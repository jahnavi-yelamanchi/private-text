# PrivateText demo design system

Source: [`design-concept.png`](design-concept.png), generated for this project and implemented without borrowing Clay branding or copy.

| Token | Value | Use |
| --- | --- | --- |
| Canvas | `#fffaf0` | Full page background |
| Ink | `#0a0a0a` | Headlines, actions, navigation |
| Peach | `#ffb084` | Source editor surface |
| Lavender | `#b8a4ed` | Redacted-output surface |
| Teal | `#1a3a3a` | Measured-inference band |
| Ochre | `#e8b94a` | Entity marker accent |
| Radius | 12px / 24px | Controls / feature surfaces |

The first viewport contains the wordmark, four-item navigation, display heading, privacy illustration, and a two-column redaction workbench. The dark teal metrics surface begins beneath it. On screens below 900px, the illustration moves below the introduction and the workbench stacks into one column. Runtime data is always sourced from the API; the initial state has no fabricated entities or metrics.
