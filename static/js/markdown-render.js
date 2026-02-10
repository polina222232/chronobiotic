/**
 * Markdown Render Module
 * Handles markdown parsing and code highlighting
 */

class MarkdownRenderer {
    constructor() {
        this.marked = window.marked;
        this.hljs = window.hljs;
        this.initMarked();
    }

    initMarked() {
        if (this.marked) {
            // Configure marked options
            this.marked.setOptions({
                highlight: (code, lang) => {
                    if (lang && this.hljs) {
                        try {
                            return this.hljs.highlight(code, { language: lang }).value;
                        } catch (err) {
                            console.warn('Highlight error:', err);
                        }
                    }
                    return code;
                },
                breaks: true,
                gfm: true,
                headerIds: true,
                mangle: false
            });
        }
    }

    render(text) {
        if (!text) return '';

        if (this.marked) {
            try {
                return this.marked.parse(text);
            } catch (err) {
                console.error('Markdown parse error:', err);
                return this.escapeHtml(text).replace(/\n/g, '<br>');
            }
        }

        // Fallback: simple formatting
        return this.simpleFormat(text);
    }

    simpleFormat(text) {
        let html = this.escapeHtml(text);
        html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        html = html.replace(/\*(.*?)\*/g, '<em>$1</em>');
        html = html.replace(/`(.*?)`/g, '<code>$1</code>');
        html = html.replace(/\n/g, '<br>');
        return html;
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    renderInline(text) {
        if (!text) return '';
        if (this.marked) {
            try {
                return this.marked.parseInline(text);
            } catch (err) {
                return this.escapeHtml(text);
            }
        }
        return this.escapeHtml(text);
    }
}

// Initialize global markdown renderer
const markdownRenderer = new MarkdownRenderer();