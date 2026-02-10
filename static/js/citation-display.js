/**
 * Citation Display Module
 * Handles citation formatting, style selection, and export functionality
 */

class CitationManager {
    constructor() {
        this.citations = [];
        this.currentStyle = 'apa';
        this.isVisible = false;

        this.initElements();
        this.bindEvents();
        this.loadStyles();
    }

    initElements() {
        this.display = document.getElementById('citationDisplay');
        this.content = document.getElementById('citationContent');
        this.styleSelect = document.getElementById('citationStyleSelect');
        this.copyAllBtn = document.getElementById('copyAllCitationsBtn');
        this.exportBtn = document.getElementById('exportCitationsBtn');
        this.closeBtn = document.getElementById('closeCitationsBtn');
        this.citationCount = document.getElementById('citationCount');
        this.exportBibTeXBtn = document.getElementById('exportBibTeXBtn');
        this.exportRISBtn = document.getElementById('exportRISBtn');
    }

    bindEvents() {
        if (this.styleSelect) {
            this.styleSelect.addEventListener('change', (e) => {
                this.currentStyle = e.target.value;
                this.updateAllCitations();
                this.saveStylePreference();
            });
        }

        if (this.copyAllBtn) {
            this.copyAllBtn.addEventListener('click', () => this.copyAllCitations());
        }

        if (this.exportBtn) {
            this.exportBtn.addEventListener('click', () => this.exportCitations('txt'));
        }

        if (this.closeBtn) {
            this.closeBtn.addEventListener('click', () => this.hide());
        }

        if (this.exportBibTeXBtn) {
            this.exportBibTeXBtn.addEventListener('click', () => this.exportBibTeX());
        }

        if (this.exportRISBtn) {
            this.exportRISBtn.addEventListener('click', () => this.exportRIS());
        }
    }

    loadStyles() {
        const savedStyle = localStorage.getItem('citationStyle');
        if (savedStyle && this.styleSelect) {
            this.currentStyle = savedStyle;
            this.styleSelect.value = savedStyle;
        }
    }

    saveStylePreference() {
        localStorage.setItem('citationStyle', this.currentStyle);
    }

    show(citations) {
        this.citations = citations;
        this.render();
        if (this.display) {
            this.display.style.display = 'block';
            this.isVisible = true;
        }
        if (this.citationCount) {
            this.citationCount.textContent = citations.length;
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
            const formattedCitation = this.formatCitation(citation, index + 1);
            const citationElement = this.createCitationElement(citation, index + 1, formattedCitation);
            this.content.appendChild(citationElement);
        });

        this.updateDisplayStyle();
    }

    updateAllCitations() {
        if (!this.content) return;

        const citationItems = this.content.querySelectorAll('.citation-item');
        citationItems.forEach((item, idx) => {
            const citationId = item.dataset.citationId;
            const citation = this.citations.find(c => c.id == citationId || c.id === parseInt(citationId));
            if (citation) {
                const formattedText = this.formatCitation(citation, idx + 1);
                const textElement = item.querySelector('.citation-text');
                if (textElement) {
                    textElement.innerHTML = formattedText;
                }
            }
        });

        this.updateDisplayStyle();
    }

    updateDisplayStyle() {
        if (this.display) {
            // Remove existing style classes
            const classes = ['citation-style-apa', 'citation-style-mla', 'citation-style-chicago',
                           'citation-style-harvard', 'citation-style-vancouver', 'citation-style-ieee',
                           'citation-style-nature', 'citation-style-elsevier', 'citation-style-springer',
                           'citation-style-russian-gost', 'citation-style-russian-gost-7-1'];
            classes.forEach(cls => this.display.classList.remove(cls));

            // Add current style class
            this.display.classList.add(`citation-style-${this.currentStyle}`);
        }
    }

    formatCitation(citation, number) {
        const style = this.currentStyle;
        const formatters = {
            apa: () => this.formatAPA(citation),
            mla: () => this.formatMLA(citation),
            chicago: () => this.formatChicago(citation),
            harvard: () => this.formatHarvard(citation),
            vancouver: () => this.formatVancouver(citation),
            ieee: () => this.formatIEEE(citation),
            nature: () => this.formatNature(citation),
            elsevier: () => this.formatElsevier(citation),
            springer: () => this.formatSpringer(citation),
            'russian-gost': () => this.formatRussianGOST(citation),
            'russian-gost-7-1': () => this.formatRussianGOST7_1(citation)
        };

        const formatter = formatters[style] || formatters.apa;
        return formatter();
    }

    formatAPA(citation) {
        const authors = this.formatAuthors(citation.authors || ['Anonymous'], 'apa');
        const year = citation.year || 'n.d.';
        const title = citation.title || 'Untitled';
        const journal = citation.journal || citation.container || '';
        const volume = citation.volume ? ` ${citation.volume}` : '';
        const issue = citation.issue ? `(${citation.issue})` : '';
        const pages = citation.pages ? `, ${citation.pages}` : '';
        const doi = citation.doi ? ` https://doi.org/${citation.doi}` : '';
        const url = citation.url && !citation.doi ? ` Retrieved from ${citation.url}` : '';

        return `${authors} (${year}). ${title}. ${journal}${volume}${issue}${pages}.${doi}${url}`;
    }

    formatMLA(citation) {
        const authors = this.formatAuthors(citation.authors || ['Anonymous'], 'mla');
        const title = citation.title || 'Untitled';
        const journal = citation.journal || citation.container || '';
        const volume = citation.volume ? ` vol. ${citation.volume}` : '';
        const issue = citation.issue ? `, no. ${citation.issue}` : '';
        const year = citation.year || 'n.d.';
        const pages = citation.pages ? `, pp. ${citation.pages}` : '';
        const doi = citation.doi ? `, doi:${citation.doi}` : '';

        return `${authors}. "${title}." ${journal}${volume}${issue} (${year})${pages}.${doi}`;
    }

    formatChicago(citation) {
        const authors = this.formatAuthors(citation.authors || ['Anonymous'], 'chicago');
        const title = citation.title || 'Untitled';
        const journal = citation.journal || citation.container || '';
        const volume = citation.volume ? ` ${citation.volume}` : '';
        const issue = citation.issue ? `, no. ${citation.issue}` : '';
        const year = citation.year ? ` (${citation.year})` : '';
        const pages = citation.pages ? `: ${citation.pages}` : '';
        const doi = citation.doi ? ` https://doi.org/${citation.doi}` : '';

        return `${authors}. "${title}." ${journal}${volume}${issue}${year}${pages}.${doi}`;
    }

    formatHarvard(citation) {
        const authors = this.formatAuthors(citation.authors || ['Anonymous'], 'harvard');
        const year = citation.year || 'n.d.';
        const title = citation.title || 'Untitled';
        const journal = citation.journal || citation.container || '';
        const volume = citation.volume ? ` ${citation.volume}` : '';
        const issue = citation.issue ? `(${citation.issue})` : '';
        const pages = citation.pages ? `: ${citation.pages}` : '';

        return `${authors} (${year}) '${title}', ${journal}${volume}${issue}${pages}.`;
    }

    formatVancouver(citation) {
        const authors = this.formatAuthors(citation.authors || ['Anonymous'], 'vancouver');
        const title = citation.title || 'Untitled';
        const journal = citation.journal || citation.container || '';
        const year = citation.year || '';
        const volume = citation.volume ? `;${citation.volume}` : '';
        const issue = citation.issue ? `(${citation.issue})` : '';
        const pages = citation.pages ? `:${citation.pages}` : '';
        const doi = citation.doi ? ` doi: ${citation.doi}` : '';

        return `${authors}. ${title}. ${journal}${year}${volume}${issue}${pages}${doi}.`;
    }

    formatIEEE(citation) {
        const authors = this.formatAuthors(citation.authors || ['Anonymous'], 'ieee');
        const title = citation.title || 'Untitled';
        const journal = citation.journal || citation.container || '';
        const volume = citation.volume ? ` vol. ${citation.volume}` : '';
        const issue = citation.issue ? `, no. ${citation.issue}` : '';
        const pages = citation.pages ? `, pp. ${citation.pages}` : '';
        const year = citation.year || '';
        const doi = citation.doi ? `, doi: ${citation.doi}` : '';

        return `${authors}, "${title}," ${journal}${volume}${issue}${pages}, ${year}.${doi}`;
    }

    formatNature(citation) {
        const authors = this.formatAuthors(citation.authors || ['Anonymous'], 'nature');
        const title = citation.title || 'Untitled';
        const journal = citation.journal || citation.container || '';
        const volume = citation.volume || '';
        const pages = citation.pages || '';
        const year = citation.year || '';
        const doi = citation.doi ? ` ${citation.doi}` : '';

        let result = `${authors} ${title}. ${journal}`;
        if (volume) result += ` ${volume}`;
        if (pages) result += ` ${pages}`;
        if (year) result += ` (${year})`;
        if (doi) result += doi;
        return result;
    }

    formatElsevier(citation) {
        const authors = this.formatAuthors(citation.authors || ['Anonymous'], 'elsevier');
        const title = citation.title || 'Untitled';
        const journal = citation.journal || citation.container || '';
        const volume = citation.volume || '';
        const issue = citation.issue ? `(${citation.issue})` : '';
        const pages = citation.pages || '';
        const year = citation.year || '';
        const doi = citation.doi ? ` https://doi.org/${citation.doi}` : '';

        return `${authors}. ${title}. ${journal}. ${year};${volume}${issue}:${pages}.${doi}`;
    }

    formatSpringer(citation) {
        const authors = this.formatAuthors(citation.authors || ['Anonymous'], 'springer');
        const title = citation.title || 'Untitled';
        const journal = citation.journal || citation.container || '';
        const volume = citation.volume ? ` ${citation.volume}` : '';
        const pages = citation.pages ? ` ${citation.pages}` : '';
        const year = citation.year || '';
        const doi = citation.doi ? ` ${citation.doi}` : '';

        return `${authors}: ${title}. ${journal}${volume},${pages} (${year})${doi}.`;
    }

    formatRussianGOST(citation) {
        const authors = this.formatAuthors(citation.authors || ['Anonymous'], 'russian');
        const title = citation.title || 'Без названия';
        const journal = citation.journal || citation.container || '';
        const year = citation.year || 'б.г.';
        const volume = citation.volume ? ` Т. ${citation.volume}` : '';
        const pages = citation.pages ? ` С. ${citation.pages}` : '';
        const doi = citation.doi ? ` DOI: ${citation.doi}` : '';

        return `${authors}. ${title} // ${journal}.${year}${volume}${pages}.${doi}`;
    }

    formatRussianGOST7_1(citation) {
        const authors = this.formatAuthors(citation.authors || ['Anonymous'], 'russian');
        const title = citation.title || 'Без названия';
        const journal = citation.journal || citation.container || '';
        const year = citation.year || 'б.г.';
        const volume = citation.volume ? `. Т. ${citation.volume}` : '';
        const pages = citation.pages ? `. С. ${citation.pages}` : '';
        const doi = citation.doi ? `. DOI: ${citation.doi}` : '';

        return `${authors}. ${title} / ${authors} // ${journal}. - ${year}${volume}${pages}${doi}.`;
    }

    formatAuthors(authors, style) {
        if (!authors || authors.length === 0) return 'Anonymous';

        const maxAuthors = style === 'apa' ? 20 : 3;

        if (authors.length === 1) {
            return authors[0];
        } else if (authors.length <= maxAuthors) {
            if (style === 'apa') {
                const last = authors.pop();
                return `${authors.join(', ')} & ${last}`;
            } else if (style === 'mla') {
                const last = authors.pop();
                return `${authors.join(', ')}, and ${last}`;
            } else if (style === 'russian') {
                const last = authors.pop();
                return `${authors.join(', ')}, ${last}`;
            } else {
                return authors.join(', ');
            }
        } else {
            if (style === 'apa') {
                return `${authors[0]} et al.`;
            } else if (style === 'mla') {
                return `${authors[0]} et al.`;
            } else {
                return `${authors[0]} et al.`;
            }
        }
    }

    createCitationElement(citation, number, formattedText) {
        const div = document.createElement('div');
        div.className = 'citation-item';
        div.dataset.citationId = citation.id || number;

        div.innerHTML = `
            <div class="citation-number">${number}.</div>
            <div class="citation-body">
                <div class="citation-text">${this.escapeHtml(formattedText)}</div>
                <div class="citation-meta">
                    <span class="citation-type">${citation.type || 'Article'}</span>
                    <span class="citation-year">${citation.year || 'n.d.'}</span>
                    ${citation.doi ? `<span class="citation-doi">DOI: <a href="https://doi.org/${citation.doi}" target="_blank">${citation.doi}</a></span>` : ''}
                </div>
                <div class="citation-actions-item">
                    <button class="copy-citation" data-id="${number}" title="Copy this citation">
                        <i class="fas fa-copy"></i>
                    </button>
                    <button class="cite-in-chat" data-id="${number}" title="Cite in chat">
                        <i class="fas fa-quote-right"></i>
                    </button>
                    ${citation.url ? `<button class="open-article" data-url="${citation.url}" title="Open article">
                        <i class="fas fa-external-link-alt"></i>
                    </button>` : ''}
                </div>
            </div>
        `;

        // Add event listeners
        const copyBtn = div.querySelector('.copy-citation');
        if (copyBtn) {
            copyBtn.addEventListener('click', () => this.copySingleCitation(formattedText));
        }

        const citeBtn = div.querySelector('.cite-in-chat');
        if (citeBtn) {
            citeBtn.addEventListener('click', () => this.citeInChat(formattedText));
        }

        const openBtn = div.querySelector('.open-article');
        if (openBtn && citation.url) {
            openBtn.addEventListener('click', () => window.open(citation.url, '_blank'));
        }

        return div;
    }

    async copySingleCitation(citationText) {
        try {
            await navigator.clipboard.writeText(citationText);
            this.showToast('Citation copied to clipboard!');
        } catch (err) {
            console.error('Failed to copy:', err);
        }
    }

    async copyAllCitations() {
        const allCitations = this.citations.map((c, i) => this.formatCitation(c, i + 1));
        const text = allCitations.join('\n\n');

        try {
            await navigator.clipboard.writeText(text);
            this.showToast(`${this.citations.length} citations copied to clipboard!`);
        } catch (err) {
            console.error('Failed to copy:', err);
        }
    }

    exportCitations(format) {
        const allCitations = this.citations.map((c, i) => this.formatCitation(c, i + 1));
        const text = allCitations.join('\n\n');

        const blob = new Blob([text], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `citations_${new Date().toISOString().slice(0, 10)}.${format}`;
        a.click();
        URL.revokeObjectURL(url);

        this.showToast('Citations exported successfully!');
    }

    exportBibTeX() {
        let bibtex = '';
        this.citations.forEach((citation, i) => {
            const id = citation.id || `ref${i + 1}`;
            bibtex += `@article{${id},\n`;
            bibtex += `  author = {${citation.authors ? citation.authors.join(' and ') : 'Anonymous'}},\n`;
            bibtex += `  title = {${citation.title || 'Untitled'}},\n`;
            bibtex += `  journal = {${citation.journal || ''}},\n`;
            bibtex += `  year = {${citation.year || 'n.d.'}},\n`;
            if (citation.volume) bibtex += `  volume = {${citation.volume}},\n`;
            if (citation.issue) bibtex += `  number = {${citation.issue}},\n`;
            if (citation.pages) bibtex += `  pages = {${citation.pages}},\n`;
            if (citation.doi) bibtex += `  doi = {${citation.doi}},\n`;
            if (citation.url) bibtex += `  url = {${citation.url}},\n`;
            bibtex += `}\n\n`;
        });

        const blob = new Blob([bibtex], { type: 'application/x-bibtex' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `citations_${new Date().toISOString().slice(0, 10)}.bib`;
        a.click();
        URL.revokeObjectURL(url);

        this.showToast('BibTeX exported successfully!');
    }

    exportRIS() {
        let ris = '';
        this.citations.forEach((citation) => {
            ris += `TY  - JOUR\n`;
            ris += `AU  - ${citation.authors ? citation.authors.join('\nAU  - ') : 'Anonymous'}\n`;
            ris += `TI  - ${citation.title || 'Untitled'}\n`;
            ris += `T2  - ${citation.journal || ''}\n`;
            ris += `PY  - ${citation.year || 'n.d.'}\n`;
            if (citation.volume) ris += `VL  - ${citation.volume}\n`;
            if (citation.issue) ris += `IS  - ${citation.issue}\n`;
            if (citation.pages) ris += `SP  - ${citation.pages.split('-')[0]}\nEP  - ${citation.pages.split('-')[1] || citation.pages.split('-')[0]}\n`;
            if (citation.doi) ris += `DO  - ${citation.doi}\n`;
            if (citation.url) ris += `UR  - ${citation.url}\n`;
            ris += `ER  - \n\n`;
        });

        const blob = new Blob([ris], { type: 'application/x-research-info-systems' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `citations_${new Date().toISOString().slice(0, 10)}.ris`;
        a.click();
        URL.revokeObjectURL(url);

        this.showToast('RIS exported successfully!');
    }

    citeInChat(citationText) {
        const event = new CustomEvent('citeInChat', { detail: { citation: citationText } });
        document.dispatchEvent(event);
        this.hide();
        this.showToast('Citation added to chat input!');
    }

    showToast(message) {
        // Create toast notification
        const toast = document.createElement('div');
        toast.className = 'citation-toast';
        toast.innerHTML = `
            <i class="fas fa-check-circle"></i>
            <span>${message}</span>
        `;
        toast.style.cssText = `
            position: fixed;
            bottom: 20px;
            left: 50%;
            transform: translateX(-50%);
            background: #28a745;
            color: white;
            padding: 12px 24px;
            border-radius: 8px;
            z-index: 10000;
            animation: fadeInOut 2s ease;
            display: flex;
            gap: 10px;
            align-items: center;
            font-size: 14px;
        `;

        document.body.appendChild(toast);

        setTimeout(() => {
            toast.remove();
        }, 2000);
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}

// Initialize citation manager
const citationManager = new CitationManager();

// Export for use in other modules
window.citationManager = citationManager;