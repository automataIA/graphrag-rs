//! Simple markdown renderer for ratatui
//!
//! Converts markdown text to ratatui `Line<'static>` values.
//! Supports: # headers, **bold**, *italic*, `code`, - bullets, > quotes, ``` code blocks.

use ratatui::{
    style::{Color, Modifier, Style},
    text::{Line, Span},
};

/// Parse a markdown string into ratatui `Line` values ready for rendering.
pub fn parse_markdown(text: &str) -> Vec<Line<'static>> {
    let mut lines = Vec::new();
    let mut in_code_block = false;

    for raw_line in text.lines() {
        // Toggle code block on ``` fence
        if raw_line.trim_start().starts_with("```") {
            in_code_block = !in_code_block;
            continue;
        }

        if in_code_block {
            lines.push(Line::from(vec![Span::styled(
                raw_line.to_owned(),
                Style::default().fg(Color::Yellow),
            )]));
            continue;
        }

        if raw_line.trim().is_empty() {
            lines.push(Line::from(""));
            continue;
        }

        // Horizontal rule (--- or ===)
        if raw_line.trim().len() >= 3
            && raw_line.trim().chars().all(|c| c == '-' || c == '=' || c == '━')
        {
            lines.push(Line::from(vec![Span::styled(
                "━".repeat(50),
                Style::default().fg(Color::DarkGray),
            )]));
            continue;
        }

        // Headers: # ## ###
        let header_level = raw_line.chars().take_while(|&c| c == '#').count();
        if header_level > 0
            && header_level <= 3
            && raw_line.as_bytes().get(header_level) == Some(&b' ')
        {
            let content = raw_line[header_level + 1..].to_owned();
            let style = header_style(header_level);
            let prefix = match header_level {
                1 => "▌ ",
                2 => "  │ ",
                _ => "    · ",
            };
            lines.push(Line::from(vec![
                Span::styled(prefix.to_owned(), style),
                Span::styled(content, style),
            ]));
            continue;
        }

        // Bullets: - item or * item
        if raw_line.starts_with("- ") || raw_line.starts_with("* ") {
            let content = &raw_line[2..];
            let mut spans =
                vec![Span::styled("  • ".to_owned(), Style::default().fg(Color::Cyan))];
            spans.extend(parse_inline(content));
            lines.push(Line::from(spans));
            continue;
        }

        // Indented bullets: "  - item"
        if (raw_line.starts_with("  - ") || raw_line.starts_with("  * "))
            && raw_line.len() > 4
        {
            let content = &raw_line[4..];
            let mut spans = vec![Span::styled(
                "    ◦ ".to_owned(),
                Style::default().fg(Color::DarkGray),
            )];
            spans.extend(parse_inline(content));
            lines.push(Line::from(spans));
            continue;
        }

        // Blockquote: > text
        if raw_line.starts_with("> ") {
            let content = &raw_line[2..];
            let style = Style::default()
                .fg(Color::DarkGray)
                .add_modifier(Modifier::ITALIC);
            lines.push(Line::from(vec![
                Span::styled(
                    " │ ".to_owned(),
                    Style::default().fg(Color::DarkGray),
                ),
                Span::styled(content.to_owned(), style),
            ]));
            continue;
        }

        // Normal text with inline formatting
        lines.push(Line::from(parse_inline(raw_line)));
    }

    lines
}

fn header_style(level: usize) -> Style {
    match level {
        1 => Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD),
        2 => Style::default()
            .fg(Color::LightBlue)
            .add_modifier(Modifier::BOLD),
        _ => Style::default()
            .fg(Color::Blue)
            .add_modifier(Modifier::BOLD),
    }
}

/// Parse inline markdown markers (**bold**, *italic*, `code`) from a string.
/// Returns `Span<'static>` values using owned strings.
pub fn parse_inline(text: &str) -> Vec<Span<'static>> {
    let mut result: Vec<Span<'static>> = Vec::new();
    let mut current = String::new();
    let chars: Vec<char> = text.chars().collect();
    let n = chars.len();
    let mut i = 0;

    while i < n {
        // **bold**
        if i + 1 < n && chars[i] == '*' && chars[i + 1] == '*' {
            if !current.is_empty() {
                result.push(Span::raw(current.clone()));
                current.clear();
            }
            let start = i + 2;
            let mut end = None;
            let mut j = start;
            while j + 1 < n {
                if chars[j] == '*' && chars[j + 1] == '*' {
                    end = Some(j);
                    break;
                }
                j += 1;
            }
            if let Some(e) = end {
                let bold_text: String = chars[start..e].iter().collect();
                result.push(Span::styled(
                    bold_text,
                    Style::default().add_modifier(Modifier::BOLD),
                ));
                i = e + 2;
            } else {
                current.push('*');
                current.push('*');
                i += 2;
            }
            continue;
        }

        // *italic* (but not **)
        if chars[i] == '*' && (i + 1 >= n || chars[i + 1] != '*') {
            if !current.is_empty() {
                result.push(Span::raw(current.clone()));
                current.clear();
            }
            let start = i + 1;
            let end = chars[start..].iter().position(|&c| c == '*');
            if let Some(e) = end {
                let italic_text: String = chars[start..start + e].iter().collect();
                result.push(Span::styled(
                    italic_text,
                    Style::default().add_modifier(Modifier::ITALIC),
                ));
                i = start + e + 1;
            } else {
                current.push('*');
                i += 1;
            }
            continue;
        }

        // `inline code`
        if chars[i] == '`' {
            if !current.is_empty() {
                result.push(Span::raw(current.clone()));
                current.clear();
            }
            let start = i + 1;
            let end = chars[start..].iter().position(|&c| c == '`');
            if let Some(e) = end {
                let code_text: String = chars[start..start + e].iter().collect();
                result.push(Span::styled(
                    code_text,
                    Style::default().fg(Color::Yellow).bg(Color::DarkGray),
                ));
                i = start + e + 1;
            } else {
                current.push('`');
                i += 1;
            }
            continue;
        }

        current.push(chars[i]);
        i += 1;
    }

    if !current.is_empty() {
        result.push(Span::raw(current));
    }

    if result.is_empty() {
        result.push(Span::raw(String::new()));
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_header() {
        let lines = parse_markdown("# Hello World");
        assert_eq!(lines.len(), 1);
    }

    #[test]
    fn test_parse_bullet() {
        let lines = parse_markdown("- item one\n- item two");
        assert_eq!(lines.len(), 2);
    }

    #[test]
    fn test_parse_code_block() {
        let lines = parse_markdown("```\ncode line\n```");
        assert_eq!(lines.len(), 1);
    }

    #[test]
    fn test_parse_inline_bold() {
        let spans = parse_inline("Hello **world**!");
        assert_eq!(spans.len(), 3);
    }

    #[test]
    fn test_parse_inline_code() {
        let spans = parse_inline("Use `cargo build` now");
        assert_eq!(spans.len(), 3);
    }
}
