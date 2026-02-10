/**
 * File Upload Module
 * Handles file uploads and processing
 */

// Ждем полной загрузки DOM
document.addEventListener('DOMContentLoaded', function() {
    console.log('DOM загружен, инициализация загрузки файлов...');

    const fileInput = document.getElementById('fileInput');
    const uploadBtn = document.getElementById('fileUploadBtn');
    const selectedFilesList = document.getElementById('selectedFilesList');
    const selectedFilesDropdown = document.getElementById('selectedFilesDropdown');

    let uploadedFiles = [];

    // Проверяем наличие элементов
    if (!uploadBtn) {
        console.error('Кнопка загрузки не найдена!');
        return;
    }

    if (!fileInput) {
        console.error('Input для файлов не найден!');
        return;
    }

    // Обработчик клика по кнопке
    uploadBtn.addEventListener('click', function(e) {
        e.preventDefault();
        e.stopPropagation();
        console.log('Клик по кнопке загрузки');

        // Просто вызываем клик по скрытому input
        fileInput.click();
    });

    // Обработчик выбора файлов
    fileInput.addEventListener('change', function(e) {
        console.log('Файлы выбраны:', e.target.files);

        if (e.target.files && e.target.files.length > 0) {
            handleFiles(e.target.files);
        }

        // Сбрасываем value чтобы можно было загрузить те же файлы снова
        fileInput.value = '';
    });

    function handleFiles(files) {
        console.log('Обработка файлов:', files.length);

        // Показываем dropdown
        if (selectedFilesDropdown) {
            selectedFilesDropdown.style.display = 'block';
        }

        // Преобразуем FileList в массив и обрабатываем каждый файл
        Array.from(files).forEach(file => {
            console.log('Файл:', file.name, file.type, (file.size / 1024).toFixed(2) + ' KB');

            // Проверка размера (макс 50MB)
            const maxSize = 50 * 1024 * 1024; // 50MB
            if (file.size > maxSize) {
                showError(`Файл ${file.name} слишком большой (макс. 50MB)`);
                return;
            }

            uploadedFiles.push(file);
            displayFile(file);
            processFile(file);
        });
    }

    function displayFile(file) {
        if (!selectedFilesList) return;

        const fileTag = document.createElement('div');
        fileTag.className = 'file-tag';

        // Выбираем иконку по типу файла
        let fileIcon = 'fa-file';
        if (file.type.startsWith('image/')) {
            fileIcon = 'fa-image';
        } else if (file.type === 'application/pdf') {
            fileIcon = 'fa-file-pdf';
        } else if (file.type.includes('word')) {
            fileIcon = 'fa-file-word';
        } else if (file.type === 'text/plain') {
            fileIcon = 'fa-file-alt';
        } else if (file.type === 'application/json') {
            fileIcon = 'fa-code';
        }

        const fileSizeKB = (file.size / 1024).toFixed(1);
        const fileName = file.name.length > 30 ? file.name.substring(0, 27) + '...' : file.name;

        fileTag.innerHTML = `
            <i class="fas ${fileIcon}"></i>
            <span title="${file.name}">${fileName}</span>
            <span style="font-size: 10px; color: #666; margin-left: auto;">${fileSizeKB} KB</span>
            <button class="remove-file" data-filename="${escapeHtml(file.name)}" type="button">&times;</button>
        `;

        const removeBtn = fileTag.querySelector('.remove-file');
        removeBtn.addEventListener('click', function(e) {
            e.stopPropagation();
            removeFile(file.name);
        });

        selectedFilesList.appendChild(fileTag);
    }

    function removeFile(filename) {
        uploadedFiles = uploadedFiles.filter(f => f.name !== filename);

        if (selectedFilesList) {
            const tags = selectedFilesList.querySelectorAll('.file-tag');
            for (let tag of tags) {
                const span = tag.querySelector('span');
                if (span && span.getAttribute('title') === filename) {
                    tag.remove();
                    break;
                }
            }

            // Скрываем dropdown если файлов нет
            if (selectedFilesList.children.length === 0 && selectedFilesDropdown) {
                selectedFilesDropdown.style.display = 'none';
            }
        }

        showToast(`Файл "${filename}" удален`);
    }

    function processFile(file) {
        // Для изображений
        if (file.type.startsWith('image/')) {
            const reader = new FileReader();
            reader.onload = function(e) {
                console.log('Изображение загружено:', file.name);
                const messageInput = document.getElementById('messageInput');
                if (messageInput) {
                    const currentValue = messageInput.value;
                    const imageInfo = `[Изображение: ${file.name}] (${(file.size / 1024).toFixed(1)} KB)`;
                    messageInput.value = currentValue ? currentValue + '\n' + imageInfo : imageInfo;
                }
                showToast(`Изображение "${file.name}" загружено!`);
            };
            reader.readAsDataURL(file);
        }
        // Для текстовых файлов
        else if (file.type === 'text/plain' || file.type === 'text/markdown' || file.type === 'text/csv' || file.type === 'application/json') {
            const reader = new FileReader();
            reader.onload = function(e) {
                const content = e.target.result;
                const messageInput = document.getElementById('messageInput');
                if (messageInput) {
                    const contextMsg = `Загружен файл: ${file.name}\n\nСодержимое:\n${content.substring(0, 500)}${content.length > 500 ? '...' : ''}`;
                    messageInput.value = contextMsg;
                }
                showToast(`Файл "${file.name}" загружен!`);
            };
            reader.onerror = function() {
                showError(`Ошибка чтения файла: ${file.name}`);
            };
            reader.readAsText(file);
        }
        // Для остальных файлов
        else {
            showToast(`Файл "${file.name}" загружен (${(file.size / 1024).toFixed(1)} KB)`);
            const messageInput = document.getElementById('messageInput');
            if (messageInput) {
                const currentValue = messageInput.value;
                const fileInfo = `[Файл: ${file.name}] (${(file.size / 1024).toFixed(1)} KB, тип: ${file.type || 'неизвестен'})`;
                messageInput.value = currentValue ? currentValue + '\n' + fileInfo : fileInfo;
            }
        }
    }

    function showToast(message) {
        // Удаляем старые тосты
        const oldToasts = document.querySelectorAll('.upload-toast');
        oldToasts.forEach(toast => toast.remove());

        const toast = document.createElement('div');
        toast.className = 'upload-toast';
        toast.innerHTML = `<i class="fas fa-check-circle"></i> ${message}`;
        toast.style.cssText = `
            position: fixed;
            bottom: 80px;
            right: 20px;
            background: #28a745;
            color: white;
            padding: 12px 20px;
            border-radius: 8px;
            z-index: 10000;
            font-family: Arial, sans-serif;
            font-size: 14px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.2);
            animation: fadeOut 2s ease;
        `;

        document.body.appendChild(toast);
        setTimeout(() => toast.remove(), 2000);
    }

    function showError(message) {
        const toast = document.createElement('div');
        toast.className = 'upload-error';
        toast.innerHTML = `<i class="fas fa-exclamation-triangle"></i> ${message}`;
        toast.style.cssText = `
            position: fixed;
            bottom: 80px;
            right: 20px;
            background: #dc3545;
            color: white;
            padding: 12px 20px;
            border-radius: 8px;
            z-index: 10000;
            font-family: Arial, sans-serif;
            font-size: 14px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.2);
            animation: fadeOut 3s ease;
        `;

        document.body.appendChild(toast);
        setTimeout(() => toast.remove(), 3000);
    }

    function escapeHtml(str) {
        return str.replace(/[&<>]/g, function(m) {
            if (m === '&') return '&amp;';
            if (m === '<') return '&lt;';
            if (m === '>') return '&gt;';
            return m;
        });
    }

    // Добавляем CSS анимацию
    const style = document.createElement('style');
    style.textContent = `
        @keyframes fadeOut {
            0% { opacity: 1; transform: translateY(0); }
            70% { opacity: 1; transform: translateY(0); }
            100% { opacity: 0; transform: translateY(-20px); }
        }

        .file-upload-btn {
            background: none;
            border: none;
            cursor: pointer;
            font-size: 20px;
            color: #007bff;
            padding: 8px 12px;
            transition: all 0.3s ease;
        }

        .file-upload-btn:hover {
            color: #0056b3;
            transform: scale(1.05);
        }

        .file-upload-wrapper {
            position: relative;
            display: inline-block;
        }

        .selected-files-dropdown {
            position: absolute;
            top: 100%;
            right: 0;
            background: white;
            border: 1px solid #ddd;
            border-radius: 8px;
            min-width: 300px;
            max-width: 400px;
            max-height: 300px;
            overflow-y: auto;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            z-index: 1000;
        }

        .selected-files-list {
            padding: 10px;
        }

        .file-tag {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 8px;
            margin-bottom: 8px;
            background: #f8f9fa;
            border-radius: 6px;
            font-size: 13px;
        }

        .file-tag i {
            color: #007bff;
            width: 20px;
        }

        .file-tag span:first-of-type {
            flex: 1;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }

        .remove-file {
            background: none;
            border: none;
            color: #dc3545;
            cursor: pointer;
            font-size: 18px;
            padding: 0 4px;
            transition: all 0.2s ease;
        }

        .remove-file:hover {
            color: #c82333;
            transform: scale(1.2);
        }
    `;
    document.head.appendChild(style);

    console.log('Модуль загрузки файлов инициализирован');
});