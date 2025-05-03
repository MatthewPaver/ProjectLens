document.addEventListener('DOMContentLoaded', () => {
	// --- State Variables ---
	let currentStep = 1;
	let selectedProject = null; // { id: 'project-id', name: 'Project Name' }
	let projectFiles = []; // Array of filenames already uploaded
	let isUploading = false;
	let isProcessing = false;
	let statusInterval = null;
	let activeTab = 'create'; // 'create' or 'select'
	let files = []; // New state for file queue

	// --- Element References ---
	const stepperSteps = document.querySelectorAll('.stepper__step');
	const step1Section = document.getElementById('step1');
	const step2Section = document.getElementById('step2');
	
	// Step 1 Elements
	const newNameInput = document.getElementById('newName');
	const newNameError = document.getElementById('newNameError');
	const createProjBtn = document.getElementById('createProjBtn');
	const createTab = document.getElementById('tab-create');
	const selectTab = document.getElementById('tab-select');
	const createContent = document.getElementById('content-create');
	const selectContent = document.getElementById('content-select');
	const existingProjectListUl = document.getElementById('existingProjectList');
	const selectProjError = document.getElementById('selectProjError');

	// Step 2 Elements
	const currentProjNameSpan = document.getElementById('currentProjName');
	const changeProjBtn = document.getElementById('changeProj');
	const dropZone = document.getElementById('dropZone');
	const fileInput = document.getElementById('fileInput');
	const fileBtn = document.getElementById('fileBtn'); // "browse" button
	const uploadBtn = document.getElementById('uploadBtn'); // New upload button ref
	const cancelBtn = document.getElementById('cancelBtn'); // New cancel button ref
	const queueEl = document.getElementById('fileQueue'); // New file queue ref
	const statusDiv = document.getElementById('status'); // Status div (used by new actions)
	const processBtn = document.getElementById('processBtn'); // Process button ref
	const processStatusDiv = document.getElementById('processStatus'); // Process status ref

	// Tooltip elements
	const helpBtn    = document.querySelector('.help-btn');
  	const tooltip    = document.getElementById('helpTooltip');

	// Modal Reference
	const successModalElement = document.getElementById('processingSuccessModal');
	let successModal = null; // To store Bootstrap modal instance

	// --- Initialisation ---
	try {
		updateUI(); // Run initial UI setup
		// Fetch existing projects for the select tab initially
		fetchExistingProjects(); 
	} catch(initError) {
	}

	// --- Functions ---
	function updateUI() {
		// Update Stepper
		stepperSteps.forEach((stepEl, index) => {
			stepEl.classList.toggle('stepper__step--active', index + 1 === currentStep);
			if (index + 1 === currentStep) {
				stepEl.setAttribute('aria-current', 'step');
			}
			else {
				stepEl.removeAttribute('aria-current');
			}
			stepEl.disabled = index + 1 > currentStep; 
		});

		// Show/Hide Sections and Tabs
		if (step1Section) step1Section.hidden = (currentStep !== 1);
		if (step2Section) step2Section.hidden = (currentStep !== 2);
		if (currentStep === 1) {
			switchTab(activeTab); // Ensure correct tab content is visible
		}

		// Update Step 1 Button State (only relevant for create tab)
		if (activeTab === 'create' && newNameInput) {
			const isValid = validateProjectName(newNameInput.value) === null;
			createProjBtn.disabled = !isValid;
		} else if (createProjBtn) {
			createProjBtn.disabled = true; // Disable create button if not on create tab
		}

		// Update Step 2 Content/Button State
		if (currentStep === 2) {
			if (currentProjNameSpan && selectedProject) {
				currentProjNameSpan.textContent = selectedProject.name;
			}
			// Update NEW action buttons
			if (uploadBtn) uploadBtn.disabled = files.length === 0 || isUploading;
			if (cancelBtn) cancelBtn.hidden   = files.length === 0 || isUploading;
			
			// Update Process button (logic depends on when it should show)
			if(processBtn){
				processBtn.style.display = (projectFiles.length > 0 && !isProcessing) ? 'inline-flex' : 'none';
				processBtn.disabled = isProcessing; 
			}
		} else {
			 if(processBtn) processBtn.style.display = 'none'; // Hide process button if not on step 2
		}
		
		// Clear status messages (only processing status now?)
		clearStatus(processStatusDiv);
	}

	function displayStatus(element, message, isError = false) {
		if (!element) return;
		element.textContent = message;
		element.classList.toggle('error', isError);
		element.classList.toggle('success', !isError && message);
	}

	function clearStatus(element) {
		if (!element) return;
		element.textContent = '';
		element.classList.remove('error', 'success');
	}

	function validateProjectName(name) {
		const trimmedName = name.trim();
		if (trimmedName.length < 2) {
			return "Project name must be at least 2 characters long.";
		}
		// Basic validation - allow letters, numbers, spaces, underscore, hyphen
		if (!/^[a-zA-Z0-9 _\-]+$/.test(trimmedName)) {
			return "Invalid characters. Use letters, numbers, spaces, _, or -.";
		}
		return null; // No error
	}

	async function handleCreateProject() {
		const name = newNameInput.value.trim();
		const validationError = validateProjectName(name);
		if (validationError) {
			displayStatus(newNameError, validationError, true);
			return;
		}
		clearStatus(newNameError);
		createProjBtn.disabled = true;
		createProjBtn.textContent = 'Creating...';

		try {
			const response = await fetch('/api/projects', {
				method: 'POST',
				headers: {
					'Content-Type': 'application/json',
				},
				body: JSON.stringify({ name: name }),
			});

			const data = await response.json();

			if (!response.ok) {
				throw new Error(data.error || `HTTP error ${response.status}`);
			}

			selectedProject = data; // Store { id: '...', name: '...' }
			currentStep = 2;
			fetchProjectFiles(); // Fetch files for the newly created project
			updateUI();

		} catch (error) {
			let userMessage = `Error creating project: ${error.message}`;
			// Check for specific duplicate error message from backend
			if (error.message && error.message.includes('already exists')) {
				const existingProjectId = error.message.match(/'([^']+)'/)?.[1]; // Extract ID if possible
				userMessage = `Project '${existingProjectId || name}' already exists.`;
				// Automatically switch to select tab
				switchTab('select');
			}
			displayStatus(newNameError, userMessage, true);
		} finally {
			createProjBtn.disabled = false;
			createProjBtn.textContent = 'Create & Continue';
		}
	}

	async function fetchProjectFiles() {
		if (!selectedProject) return;
		try {
			const response = await fetch(`/api/projects/${selectedProject.id}/files`);
			if (!response.ok) {
				throw new Error(`HTTP error ${response.status}`);
			}
			projectFiles = await response.json(); // Store server files
			updateUI(); // Update UI (mainly to show Process button if projectFiles > 0)
		} catch (error) {
			displayStatus(processStatusDiv, `Error fetching existing files: ${error.message}`, true); // Show error in process status?
			projectFiles = []; // Reset on error
			updateUI(); // Update UI
		}
	}

	// --- NEW File Handling Functions (from user snippet, with additions) ---
	// Handle file selection via browse button
	function handleFileSelect(event) {
		if (event.target.files) {
			addFiles(event.target.files);
		}
	}
	
	// Drag and Drop Handlers
	function handleDragOver(event) {
		event.preventDefault(); // Necessary to allow drop
		event.stopPropagation();
		dropZone?.classList.add('dragover');
	}

	function handleDragLeave(event) {
		event.preventDefault();
		event.stopPropagation();
		dropZone?.classList.remove('dragover');
	}

	function handleDrop(event) {
		event.preventDefault();
		event.stopPropagation();
		dropZone?.classList.remove('dragover');
		const droppedFiles = event.dataTransfer?.files;
		if (droppedFiles) {
			addFiles(droppedFiles);
		}
	}

	// Add files & render queue
	function addFiles(list) {
		let addedCount = 0;
		Array.from(list).forEach(f => {
		  // Use stricter validation if needed
		  if (!/\.(csv|xlsx)$/i.test(f.name)) {
			  return;
		  }
		  // Check for duplicates already in the queue
		  const isDuplicate = files.some(item => item.file.name === f.name && item.file.size === f.size);
		  if (isDuplicate) {
			  return;
		  }
		  
		  files.push({ file: f, progress: 0, id: Date.now() + Math.random() }); // Add a simple unique ID
		  addedCount++;
		});
		renderQueue();
		updateUI(); // Update button states based on new file list
		
		if (addedCount > 0) {
			displayStatus(statusDiv, `Ready to upload ${files.length} file(s).`, false);
		} else if (list.length > 0) {
			// Indicate if files were provided but all were skipped (duplicates/invalid)
			displayStatus(statusDiv, "Selected file(s) were invalid or duplicates.", true);
		}
	}

	// Render the file queue
	function renderQueue() {
		if (!queueEl) {
			return;
		}
		if (files.length === 0) {
			queueEl.innerHTML = '<li class="list-placeholder">No files selected yet.</li>';
		} else {
			queueEl.innerHTML = files.map((item, index) => `
				<li data-file-id="${item.id}">
					<i class="bi bi-file-earmark"></i> 
					<span class="filename" title="${item.file.name}">${item.file.name}</span>
					<span class="filesize">${formatFileSize(item.file.size)}</span> {# Optional: Add file size #}
					<button class="remove-file" data-file-id="${item.id}" aria-label="Remove ${item.file.name}">&times;</button>
					<div class="progress" style="width: ${item.progress}%"></div>
				</li>
			`).join('');
		}
		updateUI(); // Ensure buttons update after render
	}

	// Handle removing a file from the queue
	function handleRemoveFile(event) {
		if (event.target.classList.contains('remove-file')) {
			const fileIdToRemove = event.target.dataset.fileId;
			// Convert fileIdToRemove to the same type as item.id if necessary (they should both be strings/numbers from Date.now)
			files = files.filter(item => String(item.id) !== fileIdToRemove);
			renderQueue();
		}
	}
	
	// Handle clearing the entire queue
	function handleClearAll() {
		files = [];
		renderQueue();
		clearStatus(statusDiv); // Clear any previous status
		// Also reset the hidden file input value so the same file can be re-added
		if(fileInput) fileInput.value = ''; 
	}

	// Format file size utility function (add if not already present)
	function formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
    }

	async function handleUploadFiles() {
		if (files.length === 0 || isUploading) {
			return;
		}
		
		// Reset error states before upload
        files.forEach(item => item.error = false);

		isUploading = true;
		updateUI(); // Disable buttons etc.
		displayStatus(statusDiv, 'Starting upload...'); // Use main status div
		let allUploadsSuccessful = true;

		try {
			// Create a copy of the files array to track progress
			const items = files.map(file => ({
				file,
				status: 'pending',
				progress: 0
			}));

			// Update UI to show pending uploads
			updateFileList(items);

			// Process each file
			for (const item of items) {
				try {
					const form = new FormData();
					form.append('file', item.file);

					// Update status to uploading
					item.status = 'uploading';
					updateFileList(items);

					const response = await fetch(`/api/projects/${selectedProject.id}/upload`, {
						method: 'POST',
						body: form
					});

					const data = await response.json();

					if (!response.ok) {
						// Handle specific error cases
						item.status = 'error';
						item.error = data.error || 'Upload failed';
						
						// Show appropriate error message based on status code
						switch (response.status) {
							case 409:
								item.error = 'File already exists';
								break;
							case 400:
								item.error = data.error || 'Invalid file';
								break;
							case 413:
								item.error = 'File too large';
								break;
							default:
								item.error = 'Upload failed';
						}
						
						showToast('error', `Failed to upload ${item.file.name}: ${item.error}`);
					} else {
						item.status = 'complete';
						showToast('success', `Successfully uploaded ${item.file.name}`);
					}
				} catch (error) {
					item.status = 'error';
					item.error = 'Network error';
					showToast('error', `Network error while uploading ${item.file.name}`);
				}

				// Update UI after each file
				updateFileList(items);
			}

			// After all files are processed, refresh the file list
			await refreshFileList();
		} finally {
			isUploading = false;
			updateUI(); // Re-enable buttons etc.
			
			if (allUploadsSuccessful && files.length > 0) {
				displayStatus(statusDiv, 'All files uploaded successfully!', false);
				// Clear the queue visually after successful upload
				files = [];
				renderQueue();
				
				// NOW fetch server files to update projectFiles and potentially trigger Process button via updateUI
				await fetchProjectFiles(); 
				
				// THEN, if you want to auto-trigger processing:
				// handleProcessProject(); 
				
			} else if (files.length > 0) {
				displayStatus(statusDiv, 'Some files failed to upload. Please review and try again.', true);
			}
			// If no files were ever added, clear status
			else {
				clearStatus(statusDiv);
			}
		}
	}

	async function handleProcessProject() {
		if (!selectedProject || isProcessing) return;

		isProcessing = true;
		processBtn.disabled = true;
		processBtn.classList.add('processing'); 
		displayStatus(processStatusDiv, 'Initiating processing...');
		clearStatus(statusDiv); // Clear upload status messages now

		try {
			const response = await fetch(`/api/projects/${selectedProject.id}/process`, {
				method: 'POST'
			});
			const data = await response.json();

			if (!response.ok) {
				throw new Error(data.error || `HTTP error ${response.status}`);
			}

			displayStatus(processStatusDiv, data.message || 'Processing started...');
			// Start polling for status
			startStatusPolling();

		} catch (error) {
			displayStatus(processStatusDiv, `Error: ${error.message}`, true);
			isProcessing = false;
			processBtn.classList.remove('processing');
			updateUI();
		}
	}

	function startStatusPolling() {
		if (statusInterval) clearInterval(statusInterval); // Clear previous interval
		
		statusInterval = setInterval(async () => {
			if (!selectedProject) {
				stopStatusPolling();
				return;
			}
			try {
				const response = await fetch(`/api/projects/${selectedProject.id}/status`);
				if (!response.ok) {
					// Don't throw error, just log and stop polling maybe?
					stopStatusPolling(); 
					displayStatus(processStatusDiv, 'Status check failed.', true);
					return;
				}
				const statusData = await response.json();
				displayStatus(processStatusDiv, statusData.msg, statusData.state === 'error');
				
				// Stop polling if done or error
				if (statusData.state === 'done' || statusData.state === 'error') {
					stopStatusPolling();
					 // Maybe fetch outputs or update UI further on completion?
					 if(statusData.state === 'done') {
						 // SUCCESS: Add highlight and show modal
						 document.querySelector('.upload-card').classList.add('processing-complete');
						 showSuccessModal();
					 }
				}
			} catch (error) {
				displayStatus(processStatusDiv, 'Error polling status.', true);
				stopStatusPolling();
			}
		}, 3000); // Poll every 3 seconds
	}

	function stopStatusPolling() {
		if (statusInterval) {
			clearInterval(statusInterval);
			statusInterval = null;
		}
		isProcessing = false;
		processBtn.classList.remove('processing');
		updateUI(); // Re-enable button, etc.
	}
	
	function showSuccessModal() {
		if (successModalElement) {
			// Set project name in modal
			const modalProjectName = successModalElement.querySelector('#modalProjectName');
			if (modalProjectName && selectedProject) {
				modalProjectName.textContent = selectedProject.name;
			}
			
			// Initialize modal instance if not already done
			if (!successModal) {
				successModal = new bootstrap.Modal(successModalElement);
			}
			successModal.show();
		}
	}

	function goBackToStep1() {
		 // Clear state related to step 2
		 files = []; // Reset the file queue
		 projectFiles = []; // Reset server file list state
		 selectedProject = null;
		 isUploading = false;
		 stopStatusPolling(); // Stop polling if active
		 renderQueue();
		 clearStatus(statusDiv);
		 clearStatus(processStatusDiv);
		 // Remove success highlight if present
		 document.querySelector('.upload-card')?.classList.remove('processing-complete'); 

		 // Switch back to create tab by default, or select tab if preferred
		 switchTab('create'); 
		 // Fetch projects again if needed for select tab
		 if (existingProjectListUl) {
		 	fetchExistingProjects();
		 }
		 
		 currentStep = 1;
		 updateUI();
	}

	// --- New Functions for Select Existing ---
	async function fetchExistingProjects() {
		if (!existingProjectListUl) return;
		existingProjectListUl.innerHTML = '<li class="list-placeholder">Loading projects...</li>'; // Show loading state
		try {
			const response = await fetch('/api/projects');
			if (!response.ok) {
				throw new Error(`HTTP error ${response.status}`);
			}
			const projects = await response.json();
			renderExistingProjectList(projects);
		} catch (error) {
			if (existingProjectListUl) existingProjectListUl.innerHTML = '<li class="list-placeholder error">Error loading projects.</li>';
			displayStatus(selectProjError, 'Could not load projects.', true);
		}
	}

	function renderExistingProjectList(projects) {
		if (!existingProjectListUl) return;
		existingProjectListUl.innerHTML = ''; // Clear
		if (projects.length === 0) {
			existingProjectListUl.innerHTML = '<li class="list-placeholder">No existing projects found.</li>';
			return;
		}
		projects.forEach(project => {
			const li = document.createElement('li');
			const button = document.createElement('button');
			button.className = 'existing-project__item';
			button.textContent = project.name;
			button.dataset.projectId = project.id;
			button.dataset.projectName = project.name;
			button.onclick = handleExistingProjectSelect;
			li.appendChild(button);
			existingProjectListUl.appendChild(li);
		});
	}

	function handleExistingProjectSelect(event) {
		const button = event.target;
		selectedProject = {
			id: button.dataset.projectId,
			name: button.dataset.projectName
		};
		clearStatus(selectProjError);
		currentStep = 2;
		fetchProjectFiles(); // Fetch files for the selected project
		updateUI();
	}

	function switchTab(targetTab) { // targetTab is 'create' or 'select'
		activeTab = targetTab;
		if (createTab) createTab.classList.toggle('tab--active', activeTab === 'create');
		if (selectTab) selectTab.classList.toggle('tab--active', activeTab === 'select');
		if (createContent) createContent.hidden = (activeTab !== 'create');
		if (selectContent) selectContent.hidden = (activeTab !== 'select');
		if (createTab) createTab.setAttribute('aria-selected', activeTab === 'create');
		if (selectTab) selectTab.setAttribute('aria-selected', activeTab === 'select');
		
		// Fetch projects when switching to select tab if list is empty
		if (activeTab === 'select' && existingProjectListUl && existingProjectListUl.children.length <= 1) { // <=1 to account for placeholder
			fetchExistingProjects();
		}
		
		// Clear errors when switching tabs
		clearStatus(newNameError);
		clearStatus(selectProjError);
	}

	// --- Event Listeners (Setup) ---
	function setupEventListeners() {
		// --- Step 1 Listeners ---
		if (createTab) createTab.addEventListener('click', () => switchTab('create'));
		if (selectTab) selectTab.addEventListener('click', () => switchTab('select'));
		if (newNameInput) {
			newNameInput.addEventListener('input', () => {
				clearStatus(newNameError); // Clear error on typing
				const isValid = validateProjectName(newNameInput.value) === null;
				createProjBtn.disabled = !isValid;
			});
		}
		if (createProjBtn) createProjBtn.addEventListener('click', handleCreateProject);
		if (existingProjectListUl) {
			// Use event delegation for project selection
			existingProjectListUl.addEventListener('click', handleExistingProjectSelect);
		}

		// --- Step 2 Listeners ---
		if (changeProjBtn) changeProjBtn.addEventListener('click', goBackToStep1);

		// File Browsing
		if (fileBtn) fileBtn.addEventListener('click', () => fileInput?.click());
		if (fileInput) fileInput.addEventListener('change', handleFileSelect);

		// Drag and Drop
		if (dropZone) {
			dropZone.addEventListener('dragover', handleDragOver);
			dropZone.addEventListener('dragleave', handleDragLeave);
			dropZone.addEventListener('drop', handleDrop);
			// Make dropzone focusable and handle keypress for accessibility
			dropZone.addEventListener('keydown', (e) => {
				if (e.key === 'Enter' || e.key === ' ') {
					e.preventDefault();
					fileInput?.click();
				}
			});
		}

		// File Queue Remove Button (using delegation on the queue element)
		if (queueEl) queueEl.addEventListener('click', handleRemoveFile);

		// Action Buttons
		if (uploadBtn) uploadBtn.addEventListener('click', handleUploadFiles); 
		if (cancelBtn) cancelBtn.addEventListener('click', handleClearAll);
		if (processBtn) processBtn.addEventListener('click', handleProcessProject);

		// Help Tooltip
		if (helpBtn && tooltip) {
			helpBtn.addEventListener('mouseenter', showHelpTooltip);
			helpBtn.addEventListener('mouseleave', hideHelpTooltip);
			helpBtn.addEventListener('focus', showHelpTooltip);
			helpBtn.addEventListener('blur', hideHelpTooltip);
		}
		
		// Modal Listener (for when modal is hidden)
		if (successModalElement) {
			successModalElement.addEventListener('hidden.bs.modal', () => {
				// Optionally reset state or perform other actions
			});
			successModal = new bootstrap.Modal(successModalElement); // Initialise Bootstrap modal
		}
	}

	// --- Final Setup ---
});
