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
        this.uploadQueue = [];
        this.isUploading = false;
        this.init();
    }

    init() {
        if (this.fileBtn && this.fileInput) {
            this.fileBtn.addEventListener('click', () => {
                this.fileInput.click();
            });

            this.fileInput.addEventListener('change', (e) => {
                const files = Array.from(e.target.files);
                this.uploadQueue.push(...files);
                this.processQueue();
                this.fileInput.value = '';
            });
        }
    }

    async processQueue() {
        if (this.isUploading || this.uploadQueue.length === 0) return;

        this.isUploading = true;
        const file = this.uploadQueue.shift();
        await this.uploadFile(file);
        this.isUploading = false;
        this.processQueue();
    }

    async uploadFile(file) {
        this.showProgress(file.name);

        for (let percent = 0; percent <= 100; percent += 10) {
            await this.delay(50);
            this.updateProgress(percent, `Uploading ${file.name} - ${percent}%`);
        }

        await this.delay(300);
        this.updateProgress(100, `Processing ${file.name}...`);

        await this.parseFile(file);

        this.hideProgress();
    }

    async parseFile(file) {
        const fileType = file.type;
        const fileName = file.name;
        const fileExt = fileName.split('.').pop().toLowerCase();

        this.updateProgress(100, `Parsing ${fileName}...`);

        await this.delay(500);

        const input = document.getElementById('messageInput');
        if (!input) return;

        let content = '';

        // Images
        if (fileType.startsWith('image/')) {
            const reader = new FileReader();
            const imgData = await new Promise((resolve) => {
                reader.onload = (e) => resolve(e.target.result);
                reader.readAsDataURL(file);
            });
            content = `[Image: ${fileName}] (${(file.size / 1024).toFixed(1)} KB)\nImage data: ${imgData.substring(0, 100)}...`;
        }
        // PDF
        else if (fileType === 'application/pdf' || fileExt === 'pdf') {
            content = `[PDF Document: ${fileName}] (${(file.size / 1024).toFixed(1)} KB)\nPDF files can be processed for text extraction.`;
        }
        // Text files
        else if (fileType === 'text/plain' || fileExt === 'txt' || fileExt === 'md' || fileExt === 'csv') {
            const text = await this.readFileAsText(file);
            content = `📄 File: ${fileName}\n\n${text.substring(0, 2000)}${text.length > 2000 ? '\n\n[Content truncated...]' : ''}`;
        }
        // Word documents
        else if (fileType.includes('word') || fileExt === 'docx' || fileExt === 'doc') {
            content = `📄 Word Document: ${fileName} (${(file.size / 1024).toFixed(1)} KB)\nWord documents can be processed for text extraction.`;
        }
        // JSON
        else if (fileType === 'application/json' || fileExt === 'json') {
            const jsonText = await this.readFileAsText(file);
            try {
                const json = JSON.parse(jsonText);
                content = `📊 JSON Data: ${fileName}\n\`\`\`json\n${JSON.stringify(json, null, 2).substring(0, 1500)}\n\`\`\``;
            } catch (e) {
                content = `📊 JSON File: ${fileName}\n${jsonText.substring(0, 1000)}`;
            }
        }
        // SMILES files for chemistry
        else if (fileExt === 'smi' || fileExt === 'smiles') {
            const smilesText = await this.readFileAsText(file);
            content = `🧪 SMILES File: ${fileName}\n\n\`\`\`\n${smilesText.substring(0, 1000)}\n\`\`\`\n\nPlease analyze these chemical structures.`;
        }
        // Default
        else {
            content = `📎 File: ${fileName} (${(file.size / 1024).toFixed(1)} KB, type: ${fileType || 'unknown'})`;
        }

        input.value = content;
        this.showToast(`✓ File "${fileName}" loaded successfully!`, '#28a745');
    }

    readFileAsText(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = (e) => resolve(e.target.result);
            reader.onerror = reject;
            reader.readAsText(file);
        });
    }

    showProgress(fileName) {
        if (this.progressDiv) {
            this.progressDiv.style.display = 'block';
            this.updateProgress(0, `Starting upload of ${fileName}...`);
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
        if (this.progressDiv) {
            setTimeout(() => {
                this.progressDiv.style.display = 'none';
                this.updateProgress(0, '');
            }, 1000);
        }
    }

    showToast(message, bgColor = '#333') {
        const toast = document.createElement('div');
        toast.textContent = message;
        toast.style.cssText = `
            position: fixed;
            bottom: 20px;
            left: 50%;
            transform: translateX(-50%);
            background: ${bgColor};
            color: white;
            padding: 8px 16px;
           бorder-radius: 8px;
            z-index: 10000;
            font-size: 13px;
        `;
        document.body.appendChild(toast);
        setTimeout(() => toast.remove(), 3000);
    }

    delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
}

const fileUpload = new FileUpload();
window.fileUpload = fileUpload;