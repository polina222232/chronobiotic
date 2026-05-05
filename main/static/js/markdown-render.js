/**
 * Markdown Renderer - Simple markdown parsing
 */

class MarkdownRenderer {
    constructor() {
        this.useMarked = typeof marked !== 'undefined';
    }

    render(text) {
        if (!text) return '';

        if (this.useMarked) {
            try {
                return marked.parse(text);
            } catch (e) {
                console.warn('Marked parse error, using fallback:', e);
                return this.simpleRender(text);
            }
        }

        return this.simpleRender(text);
    }

    simpleRender(text) {
        let html = this.escapeHtml(text);

        // Headers
        html = html.replace(/^#### (.*$)/gm, '<h4>$1</h4>');
        html = html.replace(/^### (.*$)/gm, '<h3>$1</h3>');
        html = html.replace(/^## (.*$)/gm, '<h2>$1</h2>');
        html = html.replace(/^# (.*$)/gm, '<h1>$1</h1>');

        // Bold
        html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        html = html.replace(/__(.*?)__/g, '<strong>$1</strong>');

        // Italic
        html = html.replace(/\*(.*?)\*/g, '<em>$1</em>');
        html = html.replace(/_(.*?)_/g, '<em>$1</em>');

        // Strikethrough
        html = html.replace(/~~(.*?)~~/g, '<del>$1</del>');

        // Code blocks
        html = html.replace(/```(\w*)\n([\s\S]*?)```/g, '<pre><code class="language-$1">$2</code></pre>');

        // Inline code
        html = html.replace(/`(.*?)`/g, '<code>$1</code>');

        // Lists - unordered
        html = html.replace(/^\s*[-*+]\s+(.*$)/gm, '<li>$1</li>');
        html = html.replace(/(<li>.*<\/li>)/s, '<ul>$1</ul>');

        // Lists - ordered
        html = html.replace(/^\s*\d+\.\s+(.*$)/gm, '<li>$1</li>');
        html = html.replace(/(<li>.*<\/li>)/s, '<ol>$1</ol>');

        // Blockquotes
        html = html.replace(/^\s*>\s+(.*$)/gm, '<blockquote>$1</blockquote>');

        // Links
        html = html.replace(/\[(.*?)\]\((.*?)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');

        // Images
        html = html.replace(/!\[(.*?)\]\((.*?)\)/g, '<img src="$2" alt="$1" class="markdown-image">');

        // Horizontal rule
        html = html.replace(/^---+$/gm, '<hr>');
        html = html.replace(/^\*\*\*+$/gm, '<hr>');

        // Line breaks
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
        if (this.useMarked) {
            try {
                return marked.parseInline(text);
            } catch (e) {
                return this.escapeHtml(text);
            }
        }
        return this.escapeHtml(text);
    }
}

document.addEventListener('DOMContentLoaded', () => {
    window.markdownRenderer = new MarkdownRenderer();
});