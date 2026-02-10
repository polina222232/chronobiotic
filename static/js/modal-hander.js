/**
 * Modal Handler for Substance Information
 */

// Глобальные переменные
let currentSubstanceName = '';
let currentSmiles = '';

// Функция загрузки данных в модальное окно
function loadModalData(button) {
    // Сохраняем данные
    currentSubstanceName = button.getAttribute('data-name');
    currentSmiles = button.getAttribute('data-smiles') || 'Not specified';

    // Устанавливаем имя
    document.getElementById('modal-name').textContent = currentSubstanceName;

    // Формула - сохраняем оригинал в атрибут
    const formula = button.getAttribute('data-formula') || 'Not specified';
    const formulaElement = document.getElementById('modal-formula');
    formulaElement.setAttribute('data-original', formula);
    formulaElement.textContent = formula;

    // SMILES
    document.getElementById('modal-smiles').textContent = currentSmiles;

    // FDA статус
    const fdastatus = button.getAttribute('data-fdastatus') || 'Pending';
    const statusBadge = document.getElementById('modal-fdastatus');
    statusBadge.textContent = fdastatus;

    // Очищаем классы
    statusBadge.className = 'badge';
    if (fdastatus.includes('Approved')) {
        statusBadge.classList.add('badge-success');
    } else if (fdastatus.includes('Investigational')) {
        statusBadge.classList.add('badge-warning');
    } else {
        statusBadge.classList.add('badge-secondary');
    }

    // Описание
    const description = button.getAttribute('data-description') || 'No description available';
    document.getElementById('modal-description').textContent = description;

    // Ссылка на детали
    const linkname = button.getAttribute('data-linkname');
    document.getElementById('modal-detail-link').setAttribute('href', '/substance/' + linkname + '/');

    // Загрузка синонимов
    loadSynonyms(linkname);

    // Изображение
    const imageUrl = button.getAttribute('data-image');
    const imageElement = document.getElementById('modal-image');
    if (imageUrl && imageUrl !== 'None' && imageUrl !== '') {
        imageElement.setAttribute('src', imageUrl);
        imageElement.style.display = 'block';
    } else {
        imageElement.style.display = 'none';
    }

    // Форматируем формулу после загрузки
    setTimeout(function() {
        if (typeof formatChemicalFormula !== 'undefined') {
            updateModalFormula();
        }
    }, 50);
}

// Функция загрузки синонимов
function loadSynonyms(linkname) {
    fetch('/get_synonyms/' + linkname + '/')
        .then(response => response.json())
        .then(data => {
            const synonyms = data.synonyms.join(', ');
            document.getElementById('modal-synonyms').textContent = synonyms || 'No synonyms available';
        })
        .catch(() => {
            document.getElementById('modal-synonyms').textContent = 'No synonyms available';
        });
}

// Функция отправки вопроса в AI
function askAIAboutSubstance() {
    if (currentSubstanceName) {
        sessionStorage.setItem('pendingQuestion', 'Tell me about ' + currentSubstanceName + ' from the Chronobiotics Database');
        window.location.href = '/agent-chat/';
    }
}

// Инициализация обработчиков событий
document.addEventListener('DOMContentLoaded', function() {
    // Обработчик кнопки Ask AI
    const askAiBtn = document.getElementById('modal-ask-ai');
    if (askAiBtn) {
        askAiBtn.addEventListener('click', askAIAboutSubstance);
    }

    // Обработчик кнопки копирования SMILES
    const copyBtn = document.getElementById('copySmilesBtn');
    if (copyBtn) {
        copyBtn.addEventListener('click', function() {
            if (typeof copyToClipboard !== 'undefined') {
                copyToClipboard(currentSmiles, 'copySmilesBtn');
            }
        });
    }

    // Обработчик открытия модального окна Bootstrap
    $('#infoModal').on('show.bs.modal', function(event) {
        const button = event.relatedTarget;
        if (button) {
            loadModalData(button);
        }
    });
});