// DOM Elements
const uploadForm = document.getElementById('uploadForm');
const zipFileInput = document.getElementById('zipFile');
const fileNameDisplay = document.getElementById('fileName');
const generateBtn = document.getElementById('generateBtn');
const statusMessage = document.getElementById('statusMessage');
const downloadSection = document.getElementById('downloadSection');
const downloadBtn = document.getElementById('downloadBtn');
const spinner = document.getElementById('spinner');

// File input handling
const fileUploadWrapper = document.querySelector('.file-upload-wrapper');

fileUploadWrapper.addEventListener('click', () => {
    zipFileInput.click();
});

zipFileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
        fileNameDisplay.textContent = file.name;
        fileNameDisplay.style.color = '#667eea';
    } else {
        fileNameDisplay.textContent = 'No file chosen';
        fileNameDisplay.style.color = '#888';
    }
});

// Prevent default drag and drop
['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
    fileUploadWrapper.addEventListener(eventName, preventDefaults, false);
});

function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
}

// Highlight on drag
['dragenter', 'dragover'].forEach(eventName => {
    fileUploadWrapper.addEventListener(eventName, () => {
        fileUploadWrapper.style.borderColor = '#667eea';
        fileUploadWrapper.style.background = '#f0f0ff';
    });
});

['dragleave', 'drop'].forEach(eventName => {
    fileUploadWrapper.addEventListener(eventName, () => {
        fileUploadWrapper.style.borderColor = '#e0e0e0';
        fileUploadWrapper.style.background = '#f9f9f9';
    });
});

// Handle drop
fileUploadWrapper.addEventListener('drop', (e) => {
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        zipFileInput.files = files;
        fileNameDisplay.textContent = files[0].name;
        fileNameDisplay.style.color = '#667eea';
    }
});

// Form validation
function validateForm() {
    const numClasses = parseInt(document.getElementById('numClasses').value);
    const classNames = document.getElementById('classNames').value.trim();
    const classList = classNames.split(/\s+/).filter(name => name.length > 0);

    if (classList.length !== numClasses) {
        showStatus(
            `Number of class names (${classList.length}) doesn't match the specified number of classes (${numClasses})`,
            'error'
        );
        return false;
    }

    if (!zipFileInput.files[0]) {
        showStatus('Please select a ZIP file', 'error');
        return false;
    }

    return true;
}

// Show status message
function showStatus(message, type = 'info') {
    statusMessage.textContent = message;
    statusMessage.className = `status-message ${type}`;
    statusMessage.classList.remove('hidden');
}

// Hide status message
function hideStatus() {
    statusMessage.classList.add('hidden');
}

// Form submission
uploadForm.addEventListener('submit', async (e) => {
    e.preventDefault();

    if (!validateForm()) {
        return;
    }

    downloadSection.classList.add('hidden');
    hideStatus();

    generateBtn.disabled = true;
    generateBtn.classList.add('loading');

    showStatus('🔄 Generating dataset... This may take a few minutes.', 'info');

    const formData = new FormData();
    formData.append('zip_file', zipFileInput.files[0]);
    formData.append('output_folder', document.getElementById('outputFolder').value);
    formData.append('num_classes', document.getElementById('numClasses').value);
    formData.append('class_names', document.getElementById('classNames').value.trim());

    try {

        const response = await fetch('/api/generate', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Failed to generate dataset');
        }

        // Get the blob
        const blob = await response.blob();
        
        // Get filename from header
        let filename = 'dataset_output.zip';
        const contentDisposition = response.headers.get('content-disposition');
        if (contentDisposition) {
            const filenameMatch = contentDisposition.match(/filename[^;=\n]*=((['"]).*?\2|[^;\n]*)/);
            if (filenameMatch && filenameMatch[1]) {
                filename = filenameMatch[1].replace(/['"]/g, '');
            }
        }
        
        // Ensure .zip extension
        if (!filename.endsWith('.zip')) {
            filename = filename.replace(/\.[^.]*$/, '') + '.zip';
        }
        
        // Create temporary download link and trigger download
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        
        // Cleanup
        setTimeout(() => {
            document.body.removeChild(a);
            window.URL.revokeObjectURL(url);
        }, 100);
        
        // Show success
        hideStatus();
        downloadSection.classList.remove('hidden');
        
        // Also set the download button for manual re-download if needed
        downloadBtn.href = url;
        downloadBtn.download = filename;
        
        // Scroll to download button
        downloadSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

    } catch (error) {
        console.error('Error:', error);
        showStatus(`❌ Error: ${error.message}`, 'error');
    } finally {
        generateBtn.disabled = false;
        generateBtn.classList.remove('loading');
    }
});

// Health check on load
window.addEventListener('load', async () => {
    try {
        const response = await fetch('/api/health');
        const data = await response.json();
        console.log('✅ Server status:', data.message);
    } catch (error) {
        console.error('❌ Server not reachable:', error);
        showStatus('⚠️ Warning: Cannot connect to server', 'error');
    }
});
