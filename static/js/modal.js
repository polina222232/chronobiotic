$('#infoModal').on('show.bs.modal', function (event) {
  var button = $(event.relatedTarget);
  var id = button.data('id');

  // Очищаем содержимое перед загрузкой
  var modal = $(this);
  modal.find('#modal-name').text('Loading...');
  modal.find('#modal-formula').text('');
  modal.find('#modal-status').text('');
  modal.find('#modal-description').text('');

  // Выполняем Ajax-запрос
  $.ajax({
    url: `/get_description/${id}/`,  // URL для получения данных
    method: 'GET',
    success: function (data) {
      modal.find('#modal-name').text(data.name);
      modal.find('#modal-formula').text(data.formula);
      modal.find('#modal-status').text(data.status);
      modal.find('#modal-description').text(data.description);
    },
    error: function () {
      modal.find('#modal-name').text('Error loading data');
    }
  });
});
