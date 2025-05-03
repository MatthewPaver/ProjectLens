// CSV preview logic for files page
function showCsvPreview(filepath) {
  const modal = new bootstrap.Modal(document.getElementById('previewModal'));
  const fileNameSpan = document.getElementById('previewFileName');
  const tableHead = document.getElementById('previewTableHead');
  const tableBody = document.getElementById('previewTableBody');
  fileNameSpan.textContent = filepath.split('/').pop();
  fileNameSpan.title = filepath;
  tableHead.innerHTML = '<tr><th colspan="10" class="text-center text-muted">Loading preview...</th></tr>';
  tableBody.innerHTML = '';
  modal.show();
  fetch(`/api/preview_output/${encodeURIComponent(filepath)}`)
    .then(res => {
      if (!res.ok) throw new Error('Failed to fetch preview');
      return res.json();
    })
    .then(data => {
      if (!data || !data.rows || data.rows.length === 0) {
        tableHead.innerHTML = '';
        tableBody.innerHTML = '<tr><td colspan="10" class="text-center text-muted">No preview available</td></tr>';
        return;
      }
      // Render header
      tableHead.innerHTML = '<tr>' + data.header.map(h => `<th>${h}</th>`).join('') + '</tr>';
      // Render up to 20 rows
      tableBody.innerHTML = data.rows.slice(0, 20).map(row =>
        '<tr>' + row.map(cell => `<td>${cell}</td>`).join('') + '</tr>'
      ).join('');
    })
    .catch(() => {
      tableHead.innerHTML = '';
      tableBody.innerHTML = '<tr><td colspan="10" class="text-center text-danger">Failed to load preview</td></tr>';
    });
}
