// Scrolls `el` to the TOP of its nearest scrollable ancestor, rather than
// relying on Element.scrollIntoView - which, given block: "nearest"/"start",
// is free to satisfy itself by scrolling ANY ancestor it likes, including
// the window. That's fine for a one-off click, but for something that
// re-fires many times a second while a video plays (the "currently playing
// rally/action" highlight), letting it walk all the way up to the window is
// exactly what reads as "the browser yanking the whole page around under
// you" - especially on a layout where the scrollable list sits in a side
// panel and the page itself is also scrollable (see ResultsView's `xs`
// breakpoint, where the tab panel's own overflow is `visible` and the page
// scrolls instead). Walking up manually and stopping at the first real
// scroll container - never all the way to <body> - keeps the highlight's
// auto-scroll scoped to its own list, and does nothing at all when that
// list isn't independently scrollable (e.g. on mobile, where the page's own
// scroll position is left alone rather than fought over).
export function scrollElementToSectionTop(el: HTMLElement) {
  let container: HTMLElement | null = el.parentElement;
  while (container && container !== document.body && container !== document.documentElement) {
    const style = getComputedStyle(container);
    const scrollsY = style.overflowY === "auto" || style.overflowY === "scroll";
    if (scrollsY && container.scrollHeight > container.clientHeight) break;
    container = container.parentElement;
  }
  if (!container || container === document.body || container === document.documentElement) return;

  const offset = el.getBoundingClientRect().top - container.getBoundingClientRect().top + container.scrollTop;
  container.scrollTo({ top: offset, behavior: "smooth" });
}
