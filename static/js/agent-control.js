/**
 * Citation Display Module
 * Handles citation formatting, style selection, and export
 */

class CitationManager {
    constructor() {
        this.citations = [];
        this.currentStyle = localStorage.getItem('citationStyle') || 'apa';
        this.isVisible = false;
        this.init();
    }

    init() {
        this.display = document.getElementById('citationDisplay');
        this.content = document.getElementById('citationContent');
        this.styleSelect = document.getElementById('citationStyleSelect');
        this.copyAllBtn = document.getElementById('copyAllCitationsBtn');
        this.closeBtn = document.getElementById('closeCitationsBtn');

        if (this.styleSelect) {
            this.styleSelect.value = this.currentStyle;
            this.styleSelect.addEventListener('change', (e) => {
                this.currentStyle = e.target.value;
                localStorage.setItem('citationStyle', this.currentStyle);
                this.render();
            });
        }

        if (this.copyAllBtn) {
            this.copyAllBtn.addEventListener('click', () => this.copyAllCitations());
        }

        if (this.closeBtn) {
            this.closeBtn.addEventListener('click', () => this.hide());
        }
    }

    show(citations) {
        this.citations = citations;
        this.render();
        if (this.display) {
            this.display.style.display = 'block';
            this.isVisible = true;
        }
    }

    hide() {
        if (this.display) {
            this.display.style.display = 'none';
            this.isVisible = false;
        }
    }

    render() {
        if (!this.content) return;

        this.content.innerHTML = '';

        this.citations.forEach((citation, index) => {
            const formattedText = this.formatCitation(citation);
            const item = this.createCitationItem(citation, index + 1, formattedText);
            this.content.appendChild(item);
        });
    }

    formatCitation(citation) {
        const styles = {
            apa: () => this.formatAPA(citation),
            mla: () => this.formatMLA(citation),
            chicago: () => this.formatChicago(citation),
            harvard: () => this.formatHarvard(citation),
            vancouver: () => this.formatVancouver(citation),
            'russian-gost': () => this.formatRussianGOST(citation)
        };

        const formatter = styles[this.currentStyle] || styles.apa;
        return formatter();
    }

    formatAPA(citation) {
        const authors = this.formatAuthors(citation.authors || ['Anonymous']);
        const year = citation.year || 'n.d.';
        const title = citation.title || 'Untitled';
        const journal = citation.journal || '';
        const volume = citation.volume ? ` ${citation.volume}` : '';
        const issue = citation.issue ? `(${citation.issue})` : '';
        const pages = citation.pages ? `, ${citation.pages}` : '';
        const doi = citation.doi ? ` https://doi.org/${citation.doi}` : '';

        return `${authors} (${year}). ${title}. ${journal}${volume}${issue}${pages}.${doi}`;
    }

    formatMLA(citation) {
        const authors = this.formatAuthors(citation.authors || ['Anonymous'], 'mla');
        const title = citation.title || 'Untitled';
        const journal = citation.journal || '';
        const volume = citation.volume ? ` vol. ${citation.volume}` : '';
        const issue = citation.issue ? `, no. ${citation.issue}` : '';
        const year = citation.year || 'n.d.';
        const pages = citation.pages ? `, pp. ${citation.pages}` : '';

        return `${authors}. "${title}." ${journal}${volume}${issue} (${year})${pages}.`;
    }

    formatChicago(citation) {
        const authors = this.formatAuthors(citation.authors || ['Anonymous'], 'chicago');
        const title = citation.title || 'Untitled';
        const journal = citation.journal || '';
        const volume = citation.volume ? ` ${citation.volume}` : '';
        const issue = citation.issue ? `, no. ${citation.issue}` : '';
        const year = citation.year ? ` (${citation.year})` : '';
        const pages = citation.pages ? `: ${citation.pages}` : '';

        return `${authors}. "${title}." ${journal}${volume}${issue}${year}${pages}.`;
    }

    formatHarvard(citation) {
        const authors = this.formatAuthors(citation.authors || ['Anonymous'], 'harvard');
        const year = citation.year || 'n.d.';
        const title = citation.title || 'Untitled';
        const journal = citation.journal || '';
        const volume = citation.volume ? ` ${citation.volume}` : '';
        const issue = citation.issue ? `(${citation.issue})` : '';
        const pages = citation.pages ? `: ${citation.pages}` : '';

        return `${authors} (${year}) '${title}', ${journal}${volume}${issue}${pages}.`;
    }

    formatVancouver(citation) {
        const authors = this.formatAuthors(citation.authors || ['Anonymous'], 'vancouver');
        const title = citation.title || 'Untitled';
        const journal = citation.journal || '';
        const year = citation.year || '';
        const volume = citation.volume ? `;${citation.volume}` : '';
        const issue = citation.issue ? `(${citation.issue})` : '';
        const pages = citation.pages ? `:${citation.pages}` : '';

        return `${authors}. ${title}. ${journal}${year}${volume}${issue}${pages}.`;
    }

    formatRussianGOST(citation) {
        const authors = this.formatAuthors(citation.authors || ['Anonymous'], 'russian');
        const title = citation.title || 'Без названия';
        const journal = citation.journal || '';
        const year = citation.year || 'б.г.';
        const volume = citation.volume ? ` Т. ${citation.volume}` : '';
        const pages = citation.pages ? ` С. ${citation.pages}` : '';

        return `${authors}. ${title} // ${journal}. ${year}${volume}${pages}.`;
    }

    formatAuthors(authors, style = 'apa') {
        if (!authors || authors.length === 0) return 'Anonymous';

        if (authors.length === 1) {
            return authors[0];
        } else if (authors.length === 2) {
            if (style === 'apa') return `${authors[0]} & ${authors[1]}`;
            if (style === 'mla') return `${authors[0]} and ${authors[1]}`;
            return `${authors[0]}, ${authors[1]}`;
        } else if (authors.length <= 3) {
            const last = authors.pop();
            return `${authors.join(', ')} & ${last}`;
        } else {
            if (style === 'russian') return `${authors[0]} и др.`;
            return `${authors[0]} et al.`;
        }
    }

    createCitationItem(citation, number, formattedText) {
        const div = document.createElement('div');
        div.className = 'citation-item';

        div.innerHTML = `
            <div class="citation-number">${number}.</div>
            <div class="citation-body">
                <div class="citation-text">${this.escapeHtml(formattedText)}</div>
                <div class="citation-meta">
                    <span class="citation-type">${citation.type || 'Article'}</span>
                    <span class="citation-year">${citation.year || 'n.d.'}</span>
                </div>
                <div class="citation-actions-item">
                    <button class="copy-citation" title="Copy"><i class="fas fa-copy"></i></button>
                    ${citation.url ? `<button class="open-article" data-url="${citation.url}" title="Open"><i class="fas fa-external-link-alt"></i></button>` : ''}
                </div>
            </div>
        `;

        const copyBtn = div.querySelector('.copy-citation');
        if (copyBtn) {
            copyBtn.addEventListener('click', () => this.copyCitation(formattedText));
        }

        const openBtn = div.querySelector('.open-article');
        if (openBtn && citation.url) {
            openBtn.addEventListener('click', () => window.open(citation.url, '_blank'));
        }

        return div;
    }

    async copyCitation(text) {
        try {
            await navigator.clipboard.writeText(text);
            this.showToast('Citation copied!');
        } catch (err) {
            console.error('Copy failed:', err);
        }
    }

    async copyAllCitations() {
        const texts = this.citations.map(c => this.formatCitation(c));
        const text = texts.join('\n\n');

        try {
            await navigator.clipboard.writeText(text);
            this.showToast(`${this.citations.length} citations copied!`);
        } catch (err) {
            console.error('Copy failed:', err);
        }
    }

    showToast(message) {
        const toast = document.createElement('div');
        toast.className = 'citation-toast';
        toast.innerHTML = `<i class="fas fa-check-circle"></i> ${message}`;
        toast.style.cssText = `
            position: fixed;
            bottom: 100px;
            right: 20px;
            background: #28a745;
            color: white;
            padding: 8px 16px;
            border-radius: 8px;
            z-index: 10000;
            font-size: 14px;
        `;
        document.body.appendChild(toast);
        setTimeout(() => toast.remove(), 2000);
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}

// Initialize citation manager
const citationManager = new CitationManager();
window.citationManager = citationManager;