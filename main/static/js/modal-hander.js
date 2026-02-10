
class ModalHandler {
    constructor() {
        this.currentSubstanceName = '';
        this.currentSmiles = '';
        this.init();
    }

    init() {
        this.setupModalListeners();
    }

    setupModalListeners() {
        // Обработка показа модального окна
        $('#infoModal').on('show.bs.modal', (event) => {
            const button = $(event.relatedTarget);
            this.loadModalData(button);
        });

        // Кнопка Ask AI
        $('#modal-ask-ai').click(() => {
            this.askAIAboutSubstance();
        });

        // Кнопка копирования SMILES
        $('#copySmilesBtn').click(() => {
            this.copySmilesToClipboard();
        });
    }

    loadModalData(button) {
        // Сохраняем данные
        this.currentSubstanceName = button.data('name');
        this.currentSmiles = button.data('smiles') || 'Not specified';

        const modal = $('#infoModal');

        // Основные данные
        modal.find('#modal-name').text(button.data('name'));

        // Формула - сохраняем текст, форматирование будет в formulaFormatter
        const formula = button.data('formula') || 'Not specified';
        const formulaElement = modal.find('#modal-formula');
        formulaElement.text(formula);

        // SMILES
        modal.find('#modal-smiles').text(this.currentSmiles);

        // FDA статус
        this.updateFdaStatus(modal, button.data('fdastatus'));

        // Описание
        modal.find('#modal-description').text(
            button.data('description') || 'No description available'
        );

        // Ссылка на детали
        modal.find('#modal-detail-link').attr(
            'href',
            '/substance/' + button.data('linkname') + '/'
        );

        // Загрузка синонимов
        this.loadSynonyms(button.data('linkname'));

        // Изображение
        this.loadImage(modal, button.data('image'));

        // Форматируем формулу после загрузки
        setTimeout(() => {
            if (formulaFormatter && formulaFormatter.formatModalFormulas) {
                formulaFormatter.formatModalFormulas();
            }
        }, 50);
    }

    updateFdaStatus(modal, status) {
        const fdastatus = status || 'Pending';
        const statusBadge = modal.find('#modal-fdastatus');
        statusBadge.text(fdastatus);

        // Очищаем предыдущие классы
        statusBadge.removeClass('badge-success badge-warning badge-secondary');

        if (fdastatus.includes('Approved')) {
            statusBadge.addClass('badge-success');
        } else if (fdastatus.includes('Investigational')) {
            statusBadge.addClass('badge-warning');
        } else {
            statusBadge.addClass('badge-secondary');
        }
    }

    loadSynonyms(linkname) {
        $.get(`/get_synonyms/${linkname}/`)
            .done((data) => {
                const synonyms = data.synonyms.join(', ');
                $('#modal-synonyms').text(synonyms || 'No synonyms available');
            })
            .fail(() => {
                $('#modal-synonyms').text('No synonyms available');
            });
    }

    loadImage(modal, imageUrl) {
        const imageElement = modal.find('#modal-image');
        if (imageUrl && imageUrl !== 'None' && imageUrl !== '') {
            imageElement.attr('src', imageUrl);
            imageElement.show();
        } else {
            imageElement.hide();
        }
    }

    askAIAboutSubstance() {
        if (this.currentSubstanceName) {
            sessionStorage.setItem(
                'pendingQuestion',
                `Tell me about ${this.currentSubstanceName} from the Chronobiotics Database`
            );
            window.location.href = "/agent-chat/";
        }
    }

    async copySmilesToClipboard() {
        try {
            await navigator.clipboard.writeText(this.currentSmiles);
            const btn = $('#copySmilesBtn');
            const originalText = btn.html();
            btn.html('<i class="fas fa-check"></i> Copied!');
            setTimeout(() => btn.html(originalText), 2000);
        } catch (err) {
            console.error('Copy failed:', err);
            const btn = $('#copySmilesBtn');
            const originalText = btn.html();
            btn.html('<i class="fas fa-times"></i> Failed!');
            setTimeout(() => btn.html(originalText), 2000);
        }
    }
}

// Инициализация
const modalHandler = new ModalHandler();