"""Backend entry point, used instead of the `uvicorn` CLI.

Click (uvicorn's CLI parser) defaults to windows_expand_args=True, which
glob-expands any raw command-line argument containing "*" on Windows before
it even resolves which option that argument belongs to - so
`--reload-exclude "API/data/*"` was silently turning into one argument per
file already inside API/data, and click then rejected them as unexpected
extra arguments. Calling uvicorn.run() directly passes the pattern through
as a real Python string/list, skipping click's argv parsing altogether.
"""

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "API.main:app",
        reload=True,
        reload_dirs=["API", "Analysis"],
        reload_excludes=["API/data/*"],
        host="0.0.0.0",
        port=8000,
    )
