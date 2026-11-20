# Changelog

## 1.6.1 — 2026-11-20

- Restored the **Calls** tab: skill and MCP invocations only, with All/Skills/MCP
  filters, status, timing, search and click-to-open node inspection. Ordinary
  tool calls are no longer listed in this tab.
- Kept the host language synchronization, the macOS fullscreen click fix for the
  top panel, and the CSS parsing fix from the 1.6.0 line.
- Preserved skill/MCP call nodes when older journal pages are loaded, so the Calls
  tab still finds historical invocations.
- Isolated the overlay from host language plugins that translate visible English
  labels, keeping the tab labels in the selected plugin language.
- Updated the bundled plugin version to 1.6.1.

## 1.5.1 — 2026-11-19

- Fixed sector legend chips overlapping at narrow widths by keeping labels on one line and letting the row wrap cleanly.

## 1.5.0 — 2026-11-18

- Added a **Calls** tab for skill and MCP invocations, with All/Skills/MCP filters,
  grouped rows, status, timing, search, and click-to-open node inspection.
- Skill calls now use a hexagon glyph on the brain; MCP server calls use a dashed
  port glyph and the occipital sector. Both get family colors, labels, live HUD
  counters, hover details and call toast cards.
- Added skill/MCP identity to the node inspector and seven-language strings.
- Updated docs, package metadata and the bundled plugin version to 1.5.0.

## 1.4.1 — 2026-11-18

- Fixed the macOS fullscreen header: the title and logo no longer sit under the
  window traffic lights; the header gets a safe top inset on macOS.
- English is now the default UI language on first run. A previously chosen
  language is still remembered, and the interface no longer follows the host
  locale before the first choice.
- Refreshed the GitHub-ready package and documentation for 1.4.1.

## 1.4.0 — 2026-11-18

- Added a **History timeline**: play/pause, restart, previous/next step,
  scrubbing, 0.5×/1×/2×/4× speed, loop, and Back to live.
- Added milestone dots for prompts, thoughts, tools, answers and documents;
  clicking a dot jumps to that event.
- Playback rebuilds the brain and side panels up to the selected event and hides
  live banners/cards, so the historical state is readable.
- Text viewer now catches wheel/trackpad events directly and scrolls with the
  mouse, scrollbar, or keyboard (PageUp/PageDown after focusing the text).
- File/media viewers are opaque full-panel modals above the whole surface.
- Side panel is drag-resizable (left edge, keyboard, double-click reset) and
  remembers its width; tabs are compact enough for the wider default.

## 1.3.0 — 2026-11-16

- Added a **Files** tab for Markdown, CSV/TSV, JSON/YAML, code, logs and other
  text files from the Session workspace, with safe Markdown preview, CSV table,
  copy, download, source-node jump, and older-history paging.
- Text-file creation/access cards now pop above the brain with a file icon.
- Files are restricted to the Session workspace; secret-like names/directories are
  excluded; previews are capped at 512 KiB and rendered without raw HTML or
  remote images.
- Local media paths are now also restricted to the Session workspace.
- File/media viewers are rendered as opaque full-panel modals, so main-window
  text can no longer bleed through.
- The side panel is drag-resizable (left edge, keyboard, double-click reset) and
  remembers its width; tabs are compact enough for the wider default.

## 1.2.0 — 2026-11-16

- Added the **Media** tab: screenshots, renders and clips as thumbnails, full
  preview, source-node jump, download, and “Show earlier files”.
- Media creation/access now pops above the brain with a real thumbnail.
- Added “Show earlier files” history paging for media.
- Prompts moved to the brainstem and are quiet on the overall map; diamond shape
  retained, highlighted on hover/selection/class filter.
- Added the Classes lens and prompt anchor.

## 1.1.0 — 2026-11-15

- Added Chat/Project modes, project graph links and drill-in.
- Added RU/EN/DE/FR/ES/IT/PT UI translations and localized plugin metadata.
- Added sector guide, node hover tooltip, separate incoming/outgoing links,
  stronger selection emphasis, labels/fullscreen, chat filter, copy, and keyboard
  navigation.
- Added live-thought banner and clickable reasoning/answer/tool/document/plan/
  agent/error event cards.

## 1.0.0 — 2026-11-15

- Initial read-only 3D neural map: six brain sectors, chat selection, reasoning,
  tool calls, documents/references, model and task context, event pulses, node
  inspector, long-journal anchor/cap handling, and dark neon artwork.
