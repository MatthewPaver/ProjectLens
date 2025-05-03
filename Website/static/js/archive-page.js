document.addEventListener('DOMContentLoaded', function() {
    // DOM Elements
    const projectList = document.getElementById('projectList');
    const statusFilter = document.getElementById('statusFilter');
    const modalFileList = document.getElementById('modalFileList');
    const fileModal = new bootstrap.Modal(document.getElementById('fileModal'));
    const modalTitle = document.querySelector('.modal-title');

    // Load Projects
    async function loadProjects(status = '') {
        try {
            projectList.innerHTML = `
                <tr>
                    <td colspan="4" class="text-center">
                        <div class="spinner-border text-primary" role="status">
                            <span class="visually-hidden">Loading...</span>
                        </div>
                    </td>
                </tr>
            `;
            
            const response = await fetch('/api/archive/projects');
            if (!response.ok) throw new Error('Failed to fetch projects');
            
            const projects = await response.json();
            
            // Filter projects by status if selected
            const filteredProjects = status 
                ? projects.filter(p => p.status === status)
                : projects;

            if (!filteredProjects.length) {
                projectList.innerHTML = `
                    <tr>
                        <td colspan="4" class="text-center">No archived projects found</td>
                    </tr>
                `;
                return;
            }

            projectList.innerHTML = filteredProjects.map(project => `
                <tr>
                    <td>${escapeHtml(project.name)}</td>
                    <td>
                        <span class="badge ${project.status === 'success' ? 'bg-success' : 'bg-danger'}">
                            ${project.status === 'success' ? 'Successful' : 'Failed'}
                        </span>
                    </td>
                    <td>
                        <button class="btn btn-sm btn-outline-primary view-files-btn" 
                                data-project-id="${project.id}">
                            <i class="bi bi-folder"></i> View Files
                        </button>
                    </td>
                    <td>
                        <button class="btn btn-sm btn-primary download-all-btn" 
                                data-project-id="${project.id}">
                            <i class="bi bi-download"></i> Download All
                        </button>
                    </td>
                </tr>
            `).join('');
        } catch (error) {
            console.error('Could not load projects:', error);
            projectList.innerHTML = `
                <tr>
                    <td colspan="4" class="text-center text-danger">
                        Failed to load projects. Please try again.
                    </td>
                </tr>
            `;
        }
    }

    // Load Project Files
    async function loadProjectFiles(projectId, projectName) {
        try {
            modalTitle.textContent = `Files - ${projectName}`;
            modalFileList.innerHTML = `
                <tr>
                    <td colspan="3" class="text-center">
                        <div class="spinner-border text-primary" role="status">
                            <span class="visually-hidden">Loading...</span>
                        </div>
                    </td>
                </tr>
            `;
            
            const response = await fetch(`/api/projects/${projectId}/archive`);
            if (!response.ok) throw new Error('Failed to fetch files');
            
            const files = await response.json();
            
            if (!files.length) {
                modalFileList.innerHTML = `
                    <tr>
                        <td colspan="3" class="text-center">No files found</td>
                    </tr>
                `;
                return;
            }

            modalFileList.innerHTML = files.map(filename => `
                <tr>
                    <td>${escapeHtml(filename)}</td>
                    <td>${filename.split('.').pop().toUpperCase()}</td>
                    <td>
                        <button class="btn btn-sm btn-primary download-btn" 
                                data-filename="${filename}"
                                data-project-id="${projectId}">
                            <i class="bi bi-download"></i> Download
                        </button>
                    </td>
                </tr>
            `).join('');
        } catch (error) {
            console.error('Could not load files:', error);
            modalFileList.innerHTML = `
                <tr>
                    <td colspan="3" class="text-center text-danger">
                        Failed to load files. Please try again.
                    </td>
                </tr>
            `;
        }
    }

    // Download File
    async function downloadFile(projectId, filename) {
        try {
            const response = await fetch(`/api/download_archive/${encodeURIComponent(`${projectId}/${filename}`)}`);
            if (!response.ok) throw new Error('Download failed');

            const blob = await response.blob();
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

    // Event Listeners
    statusFilter.addEventListener('change', (e) => {
        loadProjects(e.target.value);
    });

    projectList.addEventListener('click', async (e) => {
        const viewFilesBtn = e.target.closest('.view-files-btn');
        if (viewFilesBtn) {
            const projectId = viewFilesBtn.dataset.projectId;
            const projectName = viewFilesBtn.closest('tr').querySelector('td').textContent;
            await loadProjectFiles(projectId, projectName);
            fileModal.show();
        }

        const downloadAllBtn = e.target.closest('.download-all-btn');
        if (downloadAllBtn) {
            // TODO: Implement download all functionality
            alert('Download all functionality coming soon');
        }
    });

    modalFileList.addEventListener('click', (e) => {
        const downloadBtn = e.target.closest('.download-btn');
        if (downloadBtn) {
            const { filename, projectId } = downloadBtn.dataset;
            if (filename && projectId) {
                downloadFile(projectId, filename);
            }
        }
    });

    // Escape HTML to prevent XSS
    function escapeHtml(unsafe) {
        return unsafe
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }

    // Initial Load
    loadProjects();
}); 