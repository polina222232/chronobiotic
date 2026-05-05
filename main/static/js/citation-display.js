/**
 * Citation Display - Show formatted citations with GOST support
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
        const exportBibBtn = document.getElementById('exportBibBtn');
        const exportRisBtn = document.getElementById('exportRisBtn');

        if (closeBtn) {
            closeBtn.addEventListener('click', () => this.hide());
        }

        if (copyBtn) {
            copyBtn.addEventListener('click', () => this.copyAll());
        }

        if (exportBibBtn) {
            exportBibBtn.addEventListener('click', () => this.exportBibTeX());
        }

        if (exportRisBtn) {
            exportRisBtn.addEventListener('click', () => this.exportRIS());
        }
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
        if (this.panel) {
            this.panel.style.display = 'block';
        }

        // Auto-hide after 15 seconds
        setTimeout(() => {
            if (this.panel && this.panel.style.display === 'block') {
                this.hide();
            }
        }, 15000);
    }

    hide() {
        if (this.panel) {
            this.panel.style.display = 'none';
        }
    }

    render() {
        if (!this.list) return;

        this.list.innerHTML = this.citations.map((c, i) => {
            const formatted = this.formatCitation(c);
            return `
                <div class="citation-item">
                    <div class="citation-number">${i + 1}</div>
                    <div class="citation-content">
                        <div class="citation-text">${this.escapeHtml(formatted)}</div>
                        <div class="citation-meta">
                            <span class="citation-type">${c.type || 'Article'}</span>
                            <span class="citation-year">${c.year || 'n.d.'}</span>
                            ${c.doi ? `<span class="citation-doi">DOI: ${c.doi}</span>` : ''}
                        </div>
                        <div class="citation-actions">
                            <button class="copy-citation" data-idx="${i}">📋 Copy</button>
                            ${c.url ? `<button class="open-citation" data-url="${c.url}">🔗 Open</button>` : ''}
                        </div>
                    </div>
                </div>
            `;
        }).join('');

        // Add event listeners
        this.list.querySelectorAll('.copy-citation').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const idx = parseInt(btn.dataset.idx);
                const text = this.formatCitation(this.citations[idx]);
                navigator.clipboard.writeText(text);
                this.showToast('Citation copied!');
            });
        });

        this.list.querySelectorAll('.open-citation').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const url = btn.dataset.url;
                if (url) window.open(url, '_blank');
            });
        });
    }

    formatCitation(c) {
        const styles = {
            'gost-r': () => this.formatGOST_R(c),
            'gost-7-1': () => this.formatGOST_7_1(c),
            'apa': () => this.formatAPA(c),
            'mla': () => this.formatMLA(c),
            'chicago': () => this.formatChicago(c),
            'harvard': () => this.formatHarvard(c),
            'vancouver': () => this.formatVancouver(c),
            'ieee': () => this.formatIEEE(c)
        };

        return (styles[this.currentStyle] || styles.apa)();
    }

    formatGOST_R(c) {
        const authors = this.formatAuthors(c.authors || ['Аноним']);
        const title = c.title || 'Без названия';
        const journal = c.journal || '';
        const year = c.year || 'б.г.';
        const volume = c.volume ? ` Т. ${c.volume}` : '';
        const issue = c.issue ? ` № ${c.issue}` : '';
        const pages = c.pages ? ` С. ${c.pages}` : '';
        const doi = c.doi ? ` DOI: ${c.doi}` : '';
        return `${authors}. ${title} // ${journal}.${year}${volume}${issue}${pages}.${doi}`;
    }

    formatGOST_7_1(c) {
        const authors = this.formatAuthors(c.authors || ['Аноним']);
        const title = c.title || 'Без названия';
        const journal = c.journal || '';
        const year = c.year || 'б.г.';
        const volume = c.volume ? ` Т. ${c.volume}` : '';
        const issue = c.issue ? ` Вып. ${c.issue}` : '';
        const pages = c.pages ? ` С. ${c.pages}` : '';
        return `${authors}. ${title} // ${journal}. - ${year}${volume}${issue}. - ${pages}`;
    }

    formatAPA(c) {
        const authors = this.formatAuthors(c.authors || ['Anonymous']);
        const year = c.year || 'n.d.';
        const title = c.title || 'Untitled';
        const journal = c.journal || '';
        const volume = c.volume ? ` ${c.volume}` : '';
        const issue = c.issue ? `(${c.issue})` : '';
        const pages = c.pages ? `, ${c.pages}` : '';
        const doi = c.doi ? ` https://doi.org/${c.doi}` : '';
        return `${authors} (${year}). ${title}. ${journal}${volume}${issue}${pages}.${doi}`;
    }

    formatMLA(c) {
        const authors = this.formatAuthors(c.authors || ['Anonymous']);
        const title = c.title || 'Untitled';
        const journal = c.journal || '';
        const volume = c.volume ? ` vol. ${c.volume}` : '';
        const issue = c.issue ? `, no. ${c.issue}` : '';
        const year = c.year || 'n.d.';
        const pages = c.pages ? `, pp. ${c.pages}` : '';
        return `${authors}. "${title}." ${journal}${volume}${issue} (${year})${pages}.`;
    }

    formatChicago(c) {
        const authors = this.formatAuthors(c.authors || ['Anonymous']);
        const title = c.title || 'Untitled';
        const journal = c.journal || '';
        const volume = c.volume ? ` ${c.volume}` : '';
        const issue = c.issue ? `, no. ${c.issue}` : '';
        const year = c.year ? ` (${c.year})` : '';
        const pages = c.pages ? `: ${c.pages}` : '';
        return `${authors}. "${title}." ${journal}${volume}${issue}${year}${pages}.`;
    }

    formatHarvard(c) {
        const authors = this.formatAuthors(c.authors || ['Anonymous']);
        const year = c.year || 'n.d.';
        const title = c.title || 'Untitled';
        const journal = c.journal || '';
        const volume = c.volume ? ` ${c.volume}` : '';
        const issue = c.issue ? `(${c.issue})` : '';
        const pages = c.pages ? `: ${c.pages}` : '';
        return `${authors} (${year}) '${title}', ${journal}${volume}${issue}${pages}.`;
    }

    formatVancouver(c) {
        const authors = this.formatAuthors(c.authors || ['Anonymous']);
        const title = c.title || 'Untitled';
        const journal = c.journal || '';
        const year = c.year || '';
        const volume = c.volume ? `;${c.volume}` : '';
        const issue = c.issue ? `(${c.issue})` : '';
        const pages = c.pages ? `:${c.pages}` : '';
        return `${authors}. ${title}. ${journal}${year}${volume}${issue}${pages}.`;
    }

    formatIEEE(c) {
        const authors = this.formatAuthors(c.authors || ['Anonymous']);
        const title = c.title || 'Untitled';
        const journal = c.journal || '';
        const volume = c.volume ? ` vol. ${c.volume}` : '';
        const issue = c.issue ? `, no. ${c.issue}` : '';
        const pages = c.pages ? `, pp. ${c.pages}` : '';
        const year = c.year || '';
        return `${authors}, "${title}," ${journal}${volume}${issue}${pages}, ${year}.`;
    }

    formatAuthors(authors) {
        if (!authors || authors.length === 0) return 'Anonymous';
        if (authors.length === 1) return authors[0];
        if (authors.length === 2) return `${authors[0]} & ${authors[1]}`;
        if (authors.length <= 3) return authors.join(', ');
        return `${authors[0]} et al.`;
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
            if (c.volume) bibtex += `  volume = {${c.volume}},\n`;
            if (c.issue) bibtex += `  number = {${c.issue}},\n`;
            if (c.pages) bibtex += `  pages = {${c.pages}},\n`;
            if (c.doi) bibtex += `  doi = {${c.doi}},\n`;
            bibtex += `}\n\n`;
        });

        const blob = new Blob([bibtex], { type: 'application/x-bibtex' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `citations_${new Date().toISOString().slice(0, 10)}.bib`;
        a.click();
        URL.revokeObjectURL(url);
        this.showToast('BibTeX exported!');
    }

    exportRIS() {
        let ris = '';
        this.citations.forEach((c) => {
            ris += `TY  - JOUR\n`;
            ris += `AU  - ${c.authors ? c.authors.join('\nAU  - ') : 'Anonymous'}\n`;
            ris += `TI  - ${c.title || 'Untitled'}\n`;
            ris += `T2  - ${c.journal || ''}\n`;
            ris += `PY  - ${c.year || 'n.d.'}\n`;
            if (c.volume) ris += `VL  - ${c.volume}\n`;
            if (c.issue) ris += `IS  - ${c.issue}\n`;
            if (c.pages) ris += `SP  - ${c.pages.split('-')[0]}\nEP  - ${c.pages.split('-')[1] || c.pages.split('-')[0]}\n`;
            if (c.doi) ris += `DO  - ${c.doi}\n`;
            if (c.url) ris += `UR  - ${c.url}\n`;
            ris += `ER  - \n\n`;
        });

        const blob = new Blob([ris], { type: 'application/x-research-info-systems' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `citations_${new Date().toISOString().slice(0, 10)}.ris`;
        a.click();
        URL.revokeObjectURL(url);
        this.showToast('RIS exported!');
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

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}

document.addEventListener('DOMContentLoaded', () => {
    window.citationManager = new CitationManager();
});