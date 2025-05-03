document.addEventListener('DOMContentLoaded', function() {
    const archiveFileList = document.getElementById('archiveFileList');
    let allFiles = [];
    let currentProject = '';
    let currentStatus = '';
    let currentFileSearch = '';

    function renderFiles() {
        let files = allFiles;
        if (currentProject) {
            files = files.filter(file => {
                // Project is always the second segment: success/Alpha/file.csv or failed/Alpha/file.csv
                const parts = file.split('/');
                return parts.length > 2 && parts[1].toLowerCase() === currentProject.toLowerCase();
            });
        }
        if (currentStatus) {
            files = files.filter(file => file.startsWith(currentStatus + '/'));
        }
        if (currentFileSearch) {
            const search = currentFileSearch.toLowerCase();
            files = files.filter(file => file.split('/').pop().toLowerCase().includes(search));
        }
        // Sort: Successful first, then Failed, then others
        files = files.slice().sort((a, b) => {
            const aStatus = a.startsWith('success/') ? 0 : (a.startsWith('failed/') ? 1 : 2);
            const bStatus = b.startsWith('success/') ? 0 : (b.startsWith('failed/') ? 1 : 2);
            if (aStatus !== bStatus) return aStatus - bStatus;
            return a.localeCompare(b);
        });
        // Update archive count badge
        const archiveCount = document.getElementById('archiveCount');
        if (archiveCount) {
            archiveCount.textContent = files.length + ' file' + (files.length === 1 ? '' : 's');
        }
        if (files.length === 0) {
            archiveFileList.innerHTML = `
                <tr>
                    <td colspan="5" class="text-center">No files found</td>
                </tr>
            `;
            return;
        }
        archiveFileList.innerHTML = files.map((filepath, i) => {
            const filename = filepath.split('/').pop();
            const project = filepath.split('/')[1] || '';
            const status = filepath.startsWith('success/') ? 'Successful' : (filepath.startsWith('failed/') ? 'Failed' : '');
            const fileType = filename.split('.').pop().toUpperCase();
            const statusBadge = status === 'Successful'
                ? '<span class="badge bg-success bg-gradient">Successful</span>'
                : (status === 'Failed' ? '<span class="badge bg-danger bg-gradient">Failed</span>' : '');
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
                    <td>${statusBadge}</td>
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

    async function loadArchiveFiles() {
        try {
            archiveFileList.innerHTML = `
                <tr>
                    <td colspan="5" class="text-center">
                        <div class="spinner-border text-primary" role="status">
                            <span class="visually-hidden">Loading...</span>
                        </div>
                    </td>
                </tr>
            `;
            const response = await fetch('/api/list_archive');
            if (!response.ok) throw new Error('Failed to load files');
            allFiles = await response.json();
            renderFiles();
        } catch (error) {
            console.error('Error loading files:', error);
            archiveFileList.innerHTML = `
                <tr>
                    <td colspan="5" class="text-center text-danger">
                        Error loading files
                    </td>
                </tr>
            `;
        }
    }

    // Download and preview logic
    archiveFileList.addEventListener('click', (e) => {
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
                // Show Bootstrap modal only for CSVs
                const modal = new bootstrap.Modal(document.getElementById('previewModal'));
                window.dispatchEvent(new CustomEvent('previewFile', { detail: filepath }));
                setTimeout(() => modal.show(), 50);
            } else if (filepath) {
                // Optionally show a message for unsupported types
                alert('Preview is only available for CSV files.');
            }
        }
    });
    async function downloadFile(filepath) {
        try {
            const response = await fetch(`/api/download_archive/${encodeURIComponent(filepath)}`);
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
    // Expose for sidebar filter
    window.loadArchiveFiles = function(project) {
        currentProject = project || '';
        renderFiles();
    };
    window.addEventListener('statusFilter', function(e) {
        currentStatus = e.detail || '';
        renderFiles();
    });
    window.addEventListener('fileSearch', function(e) {
        currentFileSearch = e.detail || '';
        renderFiles();
    });
    // Initial load
    loadArchiveFiles();

    // Add custom styles for alternating rows and hover
    const style = document.createElement('style');
    style.innerHTML = `
    .table-row-even { background: #f8f9fa; }
    .table-row-odd { background: #e9ecef; }
    .file-row:hover { background: #d0ebff !important; transition: background 0.2s; }
    `;
    document.head.appendChild(style);

    // Sidebar project filter logic (robust)
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
});
