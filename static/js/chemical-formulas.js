/**
 * Chemical Formulas Formatter
 * Converts chemical formulas to display with proper subscripts
 */

// Функция для форматирования химической формулы
function formatChemicalFormula(formula) {
    if (!formula || formula === 'Not specified' || formula === '') {
        return formula || 'Not specified';
    }

    let result = formula;

    // Обработка скобок с индексами (OH)2 -> (OH)<sub>2</sub>
    result = result.replace(/([\(\[][^\)\]]+[\)\]])(\d+)/g, '$1<sub>$2</sub>');

    // Обработка двухбуквенных элементов с индексами (Na, Mg, Cl и т.д.)
    result = result.replace(/([A-Z][a-z])(\d+)/g, '$1<sub>$2</sub>');

    // Обработка однобуквенных элементов с индексами (C6, H12, O6)
    result = result.replace(/([A-Z])(\d+)/g, '$1<sub>$2</sub>');

    // Обработка зарядов ионов (Fe3+ -> Fe<sup>3+</sup>)
    result = result.replace(/([A-Za-z\(\)]+)(\d+[\+\-])/g, '$1<sup>$2</sup>');

    // Обработка простых зарядов (Na+, Cl-)
    result = result.replace(/([A-Za-z\(\)]+)([\+\-])/g, '$1<sup>$2</sup>');

    return result;
}

// Функция для обновления формулы в модальном окне
function updateModalFormula() {
    const formulaElement = document.getElementById('modal-formula');
    if (formulaElement) {
        const originalText = formulaElement.getAttribute('data-original') || formulaElement.textContent;
        const formatted = formatChemicalFormula(originalText);
        if (formatted !== formulaElement.innerHTML) {
            formulaElement.innerHTML = formatted;
            formulaElement.classList.add('formula-updated');
            setTimeout(() => formulaElement.classList.remove('formula-updated'), 500);
        }
    }
}

// Функция для копирования текста
function copyToClipboard(text, elementId) {
    navigator.clipboard.writeText(text).then(function() {
        const btn = document.getElementById(elementId);
        if (btn) {
            const originalText = btn.innerHTML;
            btn.innerHTML = '<i class="fas fa-check"></i> Copied!';
            setTimeout(() => {
                btn.innerHTML = originalText;
            }, 2000);
        }
    }).catch(function(err) {
        console.error('Copy failed:', err);
        const btn = document.getElementById(elementId);
        if (btn) {
            const originalText = btn.innerHTML;
            btn.innerHTML = '<i class="fas fa-times"></i> Failed!';
            setTimeout(() => {
                btn.innerHTML = originalText;
            }, 2000);
        }
    });
}

// Ждем загрузки DOM
document.addEventListener('DOMContentLoaded', function() {
    // Наблюдатель за изменениями в модальном окне
    const observer = new MutationObserver(function(mutations) {
        mutations.forEach(function(mutation) {
            if (mutation.type === 'attributes' && mutation.attributeName === 'style') {
                setTimeout(updateModalFormula, 50);
            }
        });
    });

    const modal = document.getElementById('infoModal');
    if (modal) {
        observer.observe(modal, { attributes: true });
    }

    // Обработчик открытия модального окна Bootstrap
    $('#infoModal').on('shown.bs.modal', function() {
        setTimeout(updateModalFormula, 100);
    });
});