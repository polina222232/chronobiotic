/**
 * Citation Manager - GOST and standard citation formatting
 */

class CitationManager {
    constructor() {
        this.citations = [];
        this.currentStyle = localStorage.getItem('citationStyle') || 'gost-r';
        this.panel = document.getElementById('citationsPanel');
        this.list = document.getElementById('citationsList');
        this.init();
    }

    init() {
        const closeBtn = document.getElementById('closeCitationsBtn');
        const copyBtn = document.getElementById('copyCitationsBtn');
        const exportBtn = document.getElementById('exportBibBtn');

        if (closeBtn) closeBtn.addEventListener('click', () => this.hide());
        if (copyBtn) copyBtn.addEventListener('click', () => this.copyAll());
        if (exportBtn) exportBtn.addEventListener('click', () => this.exportBibTeX());
    }

    setStyle(style) {
        this.currentStyle = style;
        localStorage.setItem('citationStyle', style);
        if (this.citations.length > 0) {
            this.render();
        }
    }

    show(citations) {
        this.citations = citations;
        this.render();
        if (this.panel) this.panel.style.display = 'block';

        setTimeout(() => {
            if (this.panel && this.panel.style.display === 'block') {
                this.hide();
            }
        }, 15000);
    }

    hide() {
        if (this.panel) this.panel.style.display = 'none';
    }

    render() {
        if (!this.list) return;

        this.list.innerHTML = this.citations.map((c, i) => {
            const formatted = this.formatCitation(c);
            return `
                <div class="citation-item">
                    <div class="citation-number">${i + 1}</div>
                    <div class="citation-content">
                        <div class="citation-text">${formatted}</div>
                        <div class="citation-meta">
                            <span class="citation-type">${c.type || 'Article'}</span>
                            <span class="citation-year">${c.year || 'n.d.'}</span>
                        </div>
                        <div class="citation-actions">
                            <button class="copy-citation" data-idx="${i}">📋 Copy</button>
                        </div>
                    </div>
                </div>
            `;
        }).join('');

        this.list.querySelectorAll('.copy-citation').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const idx = parseInt(btn.dataset.idx);
                const text = this.formatCitation(this.citations[idx]);
                navigator.clipboard.writeText(text);
                this.showToast('Citation copied!');
            });
        });
    }

    formatCitation(c) {
        const authors = c.authors?.join(', ') || 'Anonymous';
        const year = c.year || 'n.d.';
        const title = c.title || 'Untitled';
        const journal = c.journal || '';
        return `${authors} (${year}). ${title}. ${journal}.`;
    }

    async copyAll() {
        const texts = this.citations.map(c => this.formatCitation(c));
        await navigator.clipboard.writeText(texts.join('\n\n'));
        this.showToast(`${this.citations.length} citations copied!`);
    }

    exportBibTeX() {
        let bibtex = '';
        this.citations.forEach((c, i) => {
            const id = c.doi ? c.doi.replace(/[^a-zA-Z0-9]/g, '') : `ref${i + 1}`;
            bibtex += `@article{${id},\n`;
            bibtex += `  author = {${c.authors ? c.authors.join(' and ') : 'Anonymous'}},\n`;
            bibtex += `  title = {${c.title || 'Untitled'}},\n`;
            bibtex += `  journal = {${c.journal || ''}},\n`;
            bibtex += `  year = {${c.year || 'n.d.'}},\n`;
            bibtex += `}\n\n`;
        });

        const blob = new Blob([bibtex], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `citations_${new Date().toISOString().slice(0, 10)}.bib`;
        a.click();
        URL.revokeObjectURL(url);
        this.showToast('BibTeX exported!');
    }

    showToast(message) {
        const toast = document.createElement('div');
        toast.textContent = message;
        toast.style.cssText = `
            position: fixed;
            bottom: 20px;
            left: 50%;
            transform: translateX(-50%);
            background: #28a745;
            color: white;
            padding: 6px 12px;
            border-radius: 6px;
            z-index: 10001;
            font-size: 12px;
        `;
        document.body.appendChild(toast);
        setTimeout(() => toast.remove(), 2000);
    }
}

const citationManager = new CitationManager();
window.citationManager = citationManager;