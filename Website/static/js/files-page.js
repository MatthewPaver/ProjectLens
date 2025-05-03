// CountUp animation for metrics
function animateCountUp(el, target, duration = 1200, decimals = 0) {
  const start = 0;
  const startTime = performance.now();
  function update(now) {
    const elapsed = now - startTime;
    const progress = Math.min(elapsed / duration, 1);
    const value = start + (target - start) * progress;
    el.textContent = decimals > 0 ? value.toFixed(decimals) : Math.floor(value);
    if (progress < 1) {
      requestAnimationFrame(update);
    } else {
      el.textContent = decimals > 0 ? target.toFixed(decimals) : target;
    }
  }
  requestAnimationFrame(update);
}
document.addEventListener('DOMContentLoaded', function() {
  document.querySelectorAll('.countup').forEach(el => {
    let val = el.textContent.replace(/[^\d.]/g, '');
    let decimals = (el.textContent.indexOf('.') > -1) ? 1 : 0;
    if (el.id === 'metric-accuracy') decimals = 0;
    animateCountUp(el, parseFloat(val), 1200, decimals);
  });
});

document.addEventListener('DOMContentLoaded', function() {
    const fileList = document.getElementById('fileList');
    let allFiles = [];
    let currentProject = '';
    let currentFileSearch = '';

    function renderFiles() {
        let files = allFiles;
        if (currentProject) {
            files = files.filter(file => {
                // Project is always the first segment: Alpha/file.csv
                const parts = file.split('/');
                return parts.length > 1 && parts[0].toLowerCase() === currentProject.toLowerCase();
            });
        }
        if (currentFileSearch) {
            const search = currentFileSearch.toLowerCase();
            files = files.filter(file => file.split('/').pop().toLowerCase().includes(search));
        }
        // Update file count badge
        const fileCount = document.getElementById('fileCount');
        if (fileCount) {
            fileCount.textContent = files.length + ' file' + (files.length === 1 ? '' : 's');
        }
        if (files.length === 0) {
            fileList.innerHTML = `
                <tr>
                    <td colspan="4" class="text-center">No files found</td>
                </tr>
            `;
            return;
        }
        fileList.innerHTML = files.map((filepath, i) => {
            const filename = filepath.split('/').pop();
            const project = filepath.split('/')[0] || '';
            const fileType = filename.split('.').pop().toUpperCase();
            // Truncate filename with ellipsis and tooltip
            const filenameCell = `<span style="max-width: 220px; display: inline-block; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; vertical-align: bottom;" title="${filename}">${filename}</span>`;
            const rowClass = i % 2 === 0 ? 'table-row-even' : 'table-row-odd';
            // Only show preview button for CSV files
            const isCSV = filename.toLowerCase().endsWith('.csv');
            const previewBtn = isCSV
                ? `<button class="btn btn-sm btn-outline-primary preview-btn w-100 mb-1" data-filepath="${filepath}"><i class="bi bi-eye"></i> Preview</button>`
                : '';
            // Download button always present, stacked below preview if preview exists
            const downloadBtn = `<button class="btn btn-sm btn-primary download-btn w-100" data-filepath="${filepath}"><i class="bi bi-download"></i> Download</button>`;
            return `
                <tr class="${rowClass} file-row">
                    <td>${filenameCell}</td>
                    <td>${project}</td>
                    <td>${fileType}</td>
                    <td style="min-width:110px;">
                        <div class="d-flex flex-column align-items-stretch gap-1">
                            ${previewBtn}${downloadBtn}
                        </div>
                    </td>
                </tr>
            `;
        }).join('');
    }

    async function loadOutputFiles() {
        try {
            fileList.innerHTML = `
                <tr>
                    <td colspan="4" class="text-center">
                        <div class="spinner-border text-primary" role="status">
                            <span class="visually-hidden">Loading...</span>
                        </div>
                    </td>
                </tr>
            `;
            const response = await fetch('/api/list_outputs');
            if (!response.ok) throw new Error('Failed to load files');
            allFiles = await response.json();
            renderFiles();
        } catch (error) {
            console.error('Error loading files:', error);
            fileList.innerHTML = `
                <tr>
                    <td colspan="4" class="text-center text-danger">
                        Error loading files
                    </td>
                </tr>
            `;
        }
    }

    // Sidebar project filter logic
    const sidebar = document.getElementById('projectSidebarList');
    if (sidebar) {
      sidebar.addEventListener('click', function(e) {
        const item = e.target.closest('li[data-project]');
        if (!item) return;
        sidebar.querySelectorAll('li').forEach(li => li.classList.remove('active'));
        item.classList.add('active');
        currentProject = item.getAttribute('data-project') || '';
        renderFiles();
      });
    }

    const fileSearch = document.getElementById('fileSearch');
    if (fileSearch) {
      fileSearch.addEventListener('input', function() {
        currentFileSearch = this.value;
        renderFiles();
      });
    }

    fileList.addEventListener('click', (e) => {
        const downloadBtn = e.target.closest('.download-btn');
        if (downloadBtn) {
            const filepath = downloadBtn.dataset.filepath;
            if (filepath) {
                downloadFile(filepath);
            }
        }
        const previewBtn = e.target.closest('.preview-btn');
        if (previewBtn) {
            const filepath = previewBtn.dataset.filepath;
            if (filepath && filepath.toLowerCase().endsWith('.csv')) {
                showCsvPreview(filepath);
            }
        }
    });

    async function downloadFile(filepath) {
        try {
            const response = await fetch(`/api/download_output/${encodeURIComponent(filepath)}`);
            if (!response.ok) throw new Error('Download failed');

            const blob = await response.blob();
            const filename = filepath.split('/').pop();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = filename;
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            document.body.removeChild(a);
        } catch (error) {
            console.error('Error downloading file:', error);
            alert('Failed to download file');
        }
    }

    // Initial Load
    loadOutputFiles();

    // Add custom styles for alternating rows and hover
    const style = document.createElement('style');
    style.innerHTML = `
    .table-row-even { background: #f8f9fa; }
    .table-row-odd { background: #e9ecef; }
    .file-row:hover { background: #d0ebff !important; transition: background 0.2s; }
    `;
    document.head.appendChild(style);
});