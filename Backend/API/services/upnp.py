"""A minimal UPnP IGD (Internet Gateway Device) client - stdlib only,
deliberately, rather than pulling in a compiled dependency (miniupnpc) for
what's ultimately a "nice to have" auto-port-forwarding toggle (see
share.py). Implements just enough of the protocol to discover a router,
read its public IP, and add/remove a TCP port mapping:

  1. SSDP discovery (discover_igd): a UDP multicast "is anyone out there"
     broadcast, asking specifically for an InternetGatewayDevice. A router
     with UPnP off - or one that simply doesn't support it - never answers,
     which is exactly how "not available, forward manually" gets detected;
     there's no separate error path for that case, just a timeout.
  2. Fetch + parse that device's XML description to find the WAN
     connection service's SOAP control URL.
  3. Plain HTTP POST SOAP calls to that URL for GetExternalIPAddress /
     AddPortMapping / DeletePortMapping.

Every device-description/SOAP response is real-world XML from a huge
variety of consumer router firmware, so parsing here deliberately ignores
XML namespaces (matching by local tag name only) rather than requiring an
exact namespace match - being lenient here is what makes this actually work
across different router vendors instead of just the one it was written
against.
"""
import http.client
import socket
import time
import urllib.parse
import xml.etree.ElementTree as ET
from typing import Optional

SSDP_ADDR = "239.255.255.250"
SSDP_PORT = 1900
SSDP_SEARCH_TARGET = "urn:schemas-upnp-org:device:InternetGatewayDevice:1"
SSDP_MX = 2  # seconds a responding device should randomize its reply delay over

HTTP_TIMEOUT_S = 5


def _local_tag(tag: str) -> str:
    """"{some-namespace}TagName" -> "TagName" - see the module docstring on
    why namespaces are ignored throughout this file."""
    return tag.split("}")[-1]


def local_ip() -> str:
    """This machine's own LAN IP - the standard "connect a UDP socket
    without ever sending anything" trick to ask the OS which local address
    it would route through, without needing any new dependency."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))
        return sock.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        sock.close()


def _parse_ssdp_headers(raw_text: str) -> dict[str, str]:
    headers: dict[str, str] = {}
    for line in raw_text.split("\r\n")[1:]:
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        headers[key.strip().lower()] = value.strip()
    return headers


def _find_text(element: ET.Element, local_name: str) -> Optional[str]:
    for child in element:
        if _local_tag(child.tag) == local_name:
            return child.text
    return None


def _fetch_device_description(location: str) -> Optional[dict]:
    parsed = urllib.parse.urlparse(location)
    try:
        conn = http.client.HTTPConnection(parsed.hostname, parsed.port or 80, timeout=HTTP_TIMEOUT_S)
        conn.request("GET", parsed.path or "/")
        resp = conn.getresponse()
        if resp.status != 200:
            return None
        body = resp.read()
        conn.close()
    except (OSError, http.client.HTTPException):
        return None

    try:
        root = ET.fromstring(body)
    except ET.ParseError:
        return None

    for service in root.iter():
        if _local_tag(service.tag) != "service":
            continue
        service_type = _find_text(service, "serviceType") or ""
        if "WANIPConnection" not in service_type and "WANPPPConnection" not in service_type:
            continue
        control_url_rel = _find_text(service, "controlURL")
        if not control_url_rel:
            continue
        return {"control_url": urllib.parse.urljoin(location, control_url_rel), "service_type": service_type}

    return None


def discover_igd(timeout_s: float = 3.0) -> Optional[dict]:
    """{"control_url", "service_type"} for the first responding gateway's
    WAN connection service, or None if nothing answered within timeout_s -
    that's the normal, expected outcome on a router with UPnP disabled or
    unsupported, not treated as an error condition anywhere in this module."""
    request = (
        "M-SEARCH * HTTP/1.1\r\n"
        f"HOST: {SSDP_ADDR}:{SSDP_PORT}\r\n"
        'MAN: "ssdp:discover"\r\n'
        f"MX: {SSDP_MX}\r\n"
        f"ST: {SSDP_SEARCH_TARGET}\r\n"
        "\r\n"
    ).encode("utf-8")

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(timeout_s)
    location = None
    try:
        sock.sendto(request, (SSDP_ADDR, SSDP_PORT))
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            try:
                data, _ = sock.recvfrom(8192)
            except socket.timeout:
                break
            headers = _parse_ssdp_headers(data.decode("utf-8", errors="ignore"))
            if headers.get("location"):
                location = headers["location"]
                break
    except OSError:
        return None
    finally:
        sock.close()

    if location is None:
        return None
    return _fetch_device_description(location)


def _soap_request(control_url: str, service_type: str, action: str, params: dict) -> Optional[ET.Element]:
    parsed = urllib.parse.urlparse(control_url)
    param_xml = "".join(f"<{key}>{value}</{key}>" for key, value in params.items())
    body = (
        '<?xml version="1.0"?>'
        '<s:Envelope xmlns:s="http://schemas.xmlsoap.org/soap/envelope/" '
        's:encodingStyle="http://schemas.xmlsoap.org/soap/encoding/">'
        "<s:Body>"
        f'<u:{action} xmlns:u="{service_type}">{param_xml}</u:{action}>'
        "</s:Body></s:Envelope>"
    ).encode("utf-8")

    try:
        conn = http.client.HTTPConnection(parsed.hostname, parsed.port or 80, timeout=HTTP_TIMEOUT_S)
        conn.request(
            "POST",
            parsed.path or "/",
            body=body,
            headers={
                "Content-Type": 'text/xml; charset="utf-8"',
                "SOAPAction": f'"{service_type}#{action}"',
                "Content-Length": str(len(body)),
            },
        )
        resp = conn.getresponse()
        response_body = resp.read()
        conn.close()
        # A SOAP fault (e.g. "port already mapped") still comes back as a
        # 500 with a real, parseable XML body - read it either way and let
        # the caller decide success/failure from the parsed tag, rather
        # than treating every non-200 as an unreadable network failure.
        if resp.status not in (200, 500):
            return None
    except (OSError, http.client.HTTPException):
        return None

    try:
        return ET.fromstring(response_body)
    except ET.ParseError:
        return None


def get_external_ip(control_url: str, service_type: str) -> Optional[str]:
    root = _soap_request(control_url, service_type, "GetExternalIPAddress", {})
    if root is None:
        return None
    for element in root.iter():
        if _local_tag(element.tag) == "NewExternalIPAddress":
            return element.text
    return None


def add_port_mapping(control_url: str, service_type: str, port: int, local_client_ip: str, description: str) -> bool:
    params = {
        "NewRemoteHost": "",
        "NewExternalPort": port,
        "NewProtocol": "TCP",
        "NewInternalPort": port,
        "NewInternalClient": local_client_ip,
        "NewEnabled": 1,
        "NewPortMappingDescription": description,
        "NewLeaseDuration": 0,
    }
    root = _soap_request(control_url, service_type, "AddPortMapping", params)
    if root is None:
        return False
    return any(_local_tag(el.tag) == "AddPortMappingResponse" for el in root.iter())


def delete_port_mapping(control_url: str, service_type: str, port: int) -> bool:
    params = {"NewRemoteHost": "", "NewExternalPort": port, "NewProtocol": "TCP"}
    root = _soap_request(control_url, service_type, "DeletePortMapping", params)
    if root is None:
        return False
    return any(_local_tag(el.tag) == "DeletePortMappingResponse" for el in root.iter())
