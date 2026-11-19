# Security

## Supported versions

| Version | Supported |
| --- | --- |
| 1.5.x | Yes |
| 1.4.x | Yes |
| 1.3.x and older | No |

## Reporting a vulnerability

Please use **GitHub private vulnerability reporting / Security advisories** on the
repository where this plugin is hosted. Do not include API keys, session dumps,
screenshots with secrets, or other sensitive data in a public issue.

There is no email address or telemetry endpoint in this project.

## Security model

Neural mind is a **read-only client-side observer** for DeepSeek Harness:

- It makes no outbound network requests, sends no telemetry, and adds no model
  requests or Host services.
- Session events are read through the existing `ctx.sessions` Client Session
  Controller. Durable images are read through the existing authorized
  `uiConversation.imageUrl(sessionId, attachment)` cache. Local deliverable files
  use DSH's authenticated same-origin `/api/file` route through the Session
  filesystem.
- It can see the Sessions the user can select in DSH because it uses the same
  Session Controller and attachment authorization as the main interface.
- There are no hardcoded hosts, user paths, credentials, API keys, tokens, or
  cookies in the bundle. The package is `"private": true` to prevent an accidental
  npm publish.
- The plugin does not evaluate journal content as code. Filenames, descriptions,
  text, Markdown output and table cells are rendered as React text/elements, never
  with `dangerouslySetInnerHTML` or an iframe.

### Skill and MCP calls

- The Calls tab and brain call nodes are **derived from the existing Session
  journal**. The plugin recognizes the `skill` tool and tool names beginning with
  `mcp__`, parses their visible names and arguments, and renders a local summary.
- The plugin **does not invoke skills, start MCP servers, open MCP transports,
  read MCP credentials, or make MCP requests**. It has no MCP client code and no
  network path beyond the authenticated same-origin DSH file route used by media.
- Tool arguments and results stay in the browser; they are not sent to a model,
  analytics endpoint, or third party.

### Files and media

- Media is displayed with `<img>`/`<video>`. Remote image URLs from tool content
  are not fetched. Local image/video paths are only listed when lexically inside
  the selected Session workspace.
- The Files tab lists only text files inside the Session workspace. Files and
  directories with secret-like names are excluded (`.env*`, private keys,
  keystores, credentials/token/password names, `.ssh`, `.aws`, `.gnupg`, `.dsh`,
  `.git`, `node_modules`, and similar paths).
- Text content is fetched only when the user opens a file, with a 512 KiB limit.
  Markdown is converted to React elements without raw HTML or remote images;
  links are restricted to `http(s)` and open with `noopener noreferrer`. CSV/TSV
  is parsed as untrusted text into a capped table; no formulas are executed.
- Downloading is an explicit user action. The plugin never uploads or transmits
  local file content.
- **Show earlier files** pages older Session history into the loaded window. This
  does not modify history, but it can make the main conversation load older
  journal messages too, because DSH shares one Session binding.

### Residual limitations

- The Session workspace path is checked lexically. A symlink inside the workspace
  that points outside is still governed by DSH's filesystem access policy, not by
  this plugin.
- Attachments already attached to a Session are considered user-authorized
  Session data, even if their pixels contain sensitive information.
- A user who explicitly opens a non-secret-named text file can read whatever the
  DSH file route permits. This is the same authority as opening the file in DSH.
- The plugin trusts the DSH Client Session Controller, attachment service, React,
  and the running Web UI. A compromised host UI is outside this plugin's boundary.

## What this project does not contain

- No credentials, API keys, tokens, cookies, or private keys.
- No `.env` files, DSH profile data, session journals, attachments, or user files.
- No third-party binaries, telemetry SDKs, analytics, or remote scripts.
- The icon and preview artwork are original and covered by the MIT license.
