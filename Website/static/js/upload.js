document.addEventListener('DOMContentLoaded', function() {
    // State
    let currentStep = 1;
    let selectedProject = null;
    let uploadedFiles = [];

    // DOM Elements
    const step1 = document.getElementById('step1');
    const step2 = document.getElementById('step2');
    const projectNameInput = document.getElementById('projectName');
    const projectNameError = document.getElementById('projectNameError');
    const createProjectBtn = document.getElementById('createProjectBtn');
    const projectList = document.getElementById('projectList');
    const selectedProjectName = document.getElementById('selectedProjectName');
    const changeProjectBtn = document.getElementById('changeProjectBtn');
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const fileList = document.getElementById('fileList');
    const uploadStatus = document.getElementById('uploadStatus');
    const processBtn = document.getElementById('processBtn');
    const processStatus = document.getElementById('processStatus');

    // Step Navigation
    function showStep(stepNumber) {
        currentStep = stepNumber;
        step1.style.display = stepNumber === 1 ? 'block' : 'none';
        step2.style.display = stepNumber === 2 ? 'block' : 'none';
    }

    // Project Creation
    createProjectBtn.addEventListener('click', async () => {
        const name = projectNameInput.value.trim();
        
        if (name.length < 2) {
            projectNameError.textContent = 'Project name must be at least 2 characters long';
            projectNameInput.classList.add('is-invalid');
            return;
        }

        try {
            createProjectBtn.disabled = true;
            const response = await fetch('/api/projects', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ name })
            });

            const data = await response.json();
            
            if (!response.ok) {
                throw new Error(data.error || 'Failed to create project');
            }

            selectedProject = data;
            selectedProjectName.textContent = data.name;
            showStep(2);
            updateProcessButton();

        } catch (error) {
            projectNameError.textContent = error.message;
            projectNameInput.classList.add('is-invalid');
        } finally {
            createProjectBtn.disabled = false;
        }
    });

    // Project Selection
    async function loadProjects() {
        try {
            const response = await fetch('/api/projects');
            const projects = await response.json();

            if (projects.length === 0) {
                projectList.innerHTML = '<div class="text-center p-3">No projects found</div>';
                return;
            }

            projectList.innerHTML = projects.map(project => `
                <button class="list-group-item list-group-item-action" 
                        data-project-id="${project.id}" 
                        data-project-name="${project.name}">
                    ${project.name}
                </button>
            `).join('');

        } catch (error) {
            projectList.innerHTML = '<div class="text-center p-3 text-danger">Failed to load projects</div>';
        }
    }

    projectList.addEventListener('click', (e) => {
        const button = e.target.closest('button');
        if (!button) return;

        selectedProject = {
            id: button.dataset.projectId,
            name: button.dataset.projectName
        };
        selectedProjectName.textContent = selectedProject.name;
        showStep(2);
        loadProjectFiles();
    });

    // File Handling
    function getFileIcon(filename) {
        const ext = filename.split('.').pop().toLowerCase();
        if (ext === 'csv') return 'bi-filetype-csv';
        if (ext === 'xlsx') return 'bi-filetype-xlsx';
        return 'bi-file-earmark';
    }
    function getFileBadge(filename) {
        const ext = filename.split('.').pop().toUpperCase();
        return `<span class="badge bg-primary-subtle text-primary file-badge">${ext}</span>`;
    }
    function updateFileList() {
        if (uploadedFiles.length === 0) {
            fileList.innerHTML = '<div class="col-12 text-muted text-center">No files uploaded yet</div>';
            return;
        }
        fileList.innerHTML = uploadedFiles.map((filename, idx) => `
            <div class="col-12 file-card animate__animated animate__fadeInUp">
                <i class="bi ${getFileIcon(filename)} file-icon"></i>
                <span style="flex:1;">${filename} ${getFileBadge(filename)}</span>
                <button class="remove-btn" data-index="${idx}" title="Remove"><i class="bi bi-trash"></i></button>
            </div>
        `).join('');
    }
    // Remove file from list (UI only, not server)
    fileList.addEventListener('click', (e) => {
        const btn = e.target.closest('.remove-btn');
        if (btn) {
            const idx = parseInt(btn.getAttribute('data-index'));
            if (!isNaN(idx)) {
                uploadedFiles.splice(idx, 1);
                updateFileList();
                updateProcessButton();
            }
        }
    });

    async function uploadFile(file) {
        const formData = new FormData();
        formData.append('file', file);
        uploadStatus.innerHTML = `<div>Uploading ${file.name}...</div><div class="progress-bar" id="progressBar"></div>`;
        uploadStatus.className = 'text-primary';
        try {
            const response = await fetch(`/api/projects/${selectedProject.id}/upload`, {
                method: 'POST',
                body: formData
            });
            if (!response.ok) throw new Error(`Failed to upload ${file.name}`);
            uploadedFiles.push(file.name);
            updateFileList();
            uploadStatus.innerHTML = `${file.name} uploaded successfully`;
            uploadStatus.className = 'text-success';
            updateProcessButton();
        } catch (error) {
            uploadStatus.textContent = error.message;
            uploadStatus.className = 'text-danger';
        }
    }

    async function loadProjectFiles() {
        try {
            const response = await fetch(`/api/projects/${selectedProject.id}/files`);
            if (!response.ok) throw new Error('Failed to load project files');
            
            const files = await response.json();
            uploadedFiles = files;
            updateFileList();
            updateProcessButton();
        } catch (error) {
            console.error('Error loading project files:', error);
        }
    }

    function updateProcessButton() {
        processBtn.disabled = uploadedFiles.length === 0;
    }

    // File selection via input
    fileInput.addEventListener('change', (e) => {
        Array.from(e.target.files).forEach(uploadFile);
        fileInput.value = ''; // Reset input
    });

    // Drag and drop
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('dragover');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        Array.from(e.dataTransfer.files).forEach(uploadFile);
    });

    // Click to browse
    dropZone.addEventListener('click', () => fileInput.click());

    // Process Project
    processBtn.addEventListener('click', async () => {
        if (!selectedProject) return;

        try {
            // Update UI to processing state
            processBtn.disabled = true;
            processBtn.querySelector('.spinner-border').classList.remove('d-none');
            processBtn.querySelector('.btn-text').textContent = 'Processing...';
            processStatus.textContent = 'Starting project processing...';
            processStatus.className = 'mt-2 text-center text-primary';

            // Start processing
            const response = await fetch(`/api/projects/${selectedProject.id}/process`, {
                method: 'POST'
            });

            if (!response.ok) {
                throw new Error('Failed to start processing');
            }

            // Start polling for status
            pollProcessingStatus();

        } catch (error) {
            processStatus.textContent = error.message;
            processStatus.className = 'mt-2 text-center text-danger';
            processBtn.disabled = false;
            processBtn.querySelector('.spinner-border').classList.add('d-none');
            processBtn.querySelector('.btn-text').textContent = 'Process Project';
        }
    });

    let statusPollInterval;

    async function pollProcessingStatus() {
        if (statusPollInterval) clearInterval(statusPollInterval);

        statusPollInterval = setInterval(async () => {
            try {
                const response = await fetch(`/api/projects/${selectedProject.id}/status`);
                const data = await response.json();

                processStatus.textContent = data.msg;

                if (data.state === 'done') {
                    clearInterval(statusPollInterval);
                    processBtn.disabled = false;
                    processBtn.querySelector('.spinner-border').classList.add('d-none');
                    processBtn.querySelector('.btn-text').textContent = 'Process Project';
                    processStatus.className = 'mt-2 text-center text-success';
                } else if (data.state === 'error') {
                    clearInterval(statusPollInterval);
                    processBtn.disabled = false;
                    processBtn.querySelector('.spinner-border').classList.add('d-none');
                    processBtn.querySelector('.btn-text').textContent = 'Process Project';
                    processStatus.className = 'mt-2 text-center text-danger';
                }

            } catch (error) {
                clearInterval(statusPollInterval);
                processStatus.textContent = 'Error checking processing status';
                processStatus.className = 'mt-2 text-center text-danger';
                processBtn.disabled = false;
                processBtn.querySelector('.spinner-border').classList.add('d-none');
                processBtn.querySelector('.btn-text').textContent = 'Process Project';
            }
        }, 3000); // Poll every 3 seconds
    }

    // Change Project Button
    changeProjectBtn.addEventListener('click', () => {
        selectedProject = null;
        uploadedFiles = [];
        updateFileList();
        showStep(1);
        if (statusPollInterval) clearInterval(statusPollInterval);
    });

    // Initial setup
    showStep(1);
    loadProjects();
});