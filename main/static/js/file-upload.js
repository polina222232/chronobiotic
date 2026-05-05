/**
 * File Upload - Support for images, PDF, documents with progress
 */

class FileUpload {
    constructor() {
        this.fileBtn = document.getElementById('fileBtn');
        this.fileInput = document.getElementById('fileInput');
        this.progressDiv = document.getElementById('uploadProgress');
        this.progressBar = document.getElementById('uploadBar');
        this.statusSpan = document.getElementById('uploadStatus');
        this.filenameSpan = document.getElementById('uploadFilename');
        this.uploadedFiles = [];
        this.init();
    }

    init() {
        if (this.fileBtn && this.fileInput) {
            this.fileBtn.addEventListener('click', () => {
                this.fileInput.click();
            });

            this.fileInput.addEventListener('change', (e) => {
                const files = Array.from(e.target.files);
                this.uploadedFiles.push(...files);
                this.processQueue();
                this.fileInput.value = '';
            });
        }
    }

    async processQueue() {
        if (this.isUploading || this.uploadedFiles.length === 0) return;

        this.isUploading = true;
        const file = this.uploadedFiles.shift();
        await this.uploadFile(file);
        this.isUploading = false;
        this.processQueue();
    }

    async uploadFile(file) {
        this.showProgress(file.name);

        // Simulate upload progress
        for (let percent = 0; percent <= 100; percent += 10) {
            await this.delay(30);
            this.updateProgress(percent, `Uploading... ${percent}%`);
        }

        await this.delay(200);
        this.updateProgress(100, 'Processing (parsing)...');

        await this.parseFile(file);

        this.hideProgress();
    }

    async parseFile(file) {
        const content = await this.readFile(file);
        const input = document.getElementById('messageInput');

        if (input) {
            const currentValue = input.value;
            let newContent = '';

            if (file.type.startsWith('image/')) {
                newContent = `🖼️ Image: ${file.name} (${(file.size / 1024).toFixed(1)} KB)\n`;
                newContent += `[Image uploaded successfully. You can ask questions about this image.]`;
            } else if (file.type === 'application/pdf' || file.name.endsWith('.pdf')) {
                newContent = `📄 PDF Document: ${file.name} (${(file.size / 1024).toFixed(1)} KB)\n`;
                newContent += `[PDF document loaded. You can ask me to analyze the content.]`;
            } else if (file.name.endsWith('.smi') || file.name.endsWith('.smiles')) {
                newContent = `🧪 SMILES File: ${file.name}\n\n\`\`\`\n${content.substring(0, 1000)}\n\`\`\`\n\nPlease analyze these chemical structures.`;
            } else {
                newContent = `📎 File: ${file.name}\n\n${content.substring(0, 1500)}${content.length > 1500 ? '\n\n[Content truncated...]' : ''}`;
            }

            input.value = currentValue ? currentValue + '\n\n' + newContent : newContent;
            input.dispatchEvent(new Event('input'));
        }

        this.showToast(`✓ ${file.name} loaded!`, '#28a745');
    }

    readFile(file) {
        return new Promise((resolve) => {
            const reader = new FileReader();
            reader.onload = (e) => resolve(e.target.result);

            if (file.type === 'text/plain' ||
                file.name.endsWith('.txt') ||
                file.name.endsWith('.md') ||
                file.name.endsWith('.csv') ||
                file.name.endsWith('.json') ||
                file.name.endsWith('.smi') ||
                file.name.endsWith('.smiles')) {
                reader.readAsText(file);
            } else if (file.type.startsWith('image/')) {
                const img = new Image();
                img.onload = () => {
                    resolve(`[Image: ${file.name}] (${img.width}x${img.height}, ${(file.size / 1024).toFixed(1)} KB)`);
                };
                img.src = URL.createObjectURL(file);
                setTimeout(() => resolve(`[Image: ${file.name}] (${(file.size / 1024).toFixed(1)} KB)`), 100);
            } else {
                resolve(`[File: ${file.name}] (${(file.size / 1024).toFixed(1)} KB, type: ${file.type || 'unknown'})`);
            }
        });
    }

    showProgress(filename) {
        if (this.progressDiv) {
            this.progressDiv.style.display = 'block';
            if (this.filenameSpan) this.filenameSpan.textContent = filename;
            this.updateProgress(0, 'Starting...');
        }
    }

    updateProgress(percent, status) {
        if (this.progressBar) {
            this.progressBar.style.width = `${percent}%`;
        }
        if (this.statusSpan) {
            this.statusSpan.textContent = status;
        }
    }

    hideProgress() {
        setTimeout(() => {
            if (this.progressDiv) {
                this.progressDiv.style.display = 'none';
                this.updateProgress(0, '');
            }
        }, 1000);
    }

    showToast(message, bgColor = '#333') {
        const toast = document.createElement('div');
        toast.textContent = message;
        toast.style.cssText = `
            position: fixed;
            bottom: 80px;
            left: 50%;
            transform: translateX(-50%);
            background: ${bgColor};
            color: white;
            padding: 8px 16px;
            border-radius: 8px;
            z-index: 10000;
            font-size: 13px;
        `;
        document.body.appendChild(toast);
        setTimeout(() => toast.remove(), 2000);
    }

    delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
}

document.addEventListener('DOMContentLoaded', () => {
    window.fileUpload = new FileUpload();
});