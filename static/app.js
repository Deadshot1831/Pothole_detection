document.addEventListener('DOMContentLoaded', () => {
    const uploadArea = document.getElementById('upload-section');
    const videoUpload = document.getElementById('video-upload');
    const browseBtn = document.getElementById('browse-btn');
    
    const loadingSection = document.getElementById('loading-section');
    const resultsSection = document.getElementById('results-section');
    const resetBtn = document.getElementById('reset-btn');
    
    // UI interactions
    browseBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        videoUpload.click();
    });
    
    uploadArea.addEventListener('click', () => {
        videoUpload.click();
    });
    
    // Drag and Drop
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        uploadArea.addEventListener(eventName, preventDefaults, false);
    });
    
    function preventDefaults (e) {
        e.preventDefault();
        e.stopPropagation();
    }
    
    ['dragenter', 'dragover'].forEach(eventName => {
        uploadArea.addEventListener(eventName, () => {
            uploadArea.classList.add('dragover');
        }, false);
    });
    
    ['dragleave', 'drop'].forEach(eventName => {
        uploadArea.addEventListener(eventName, () => {
            uploadArea.classList.remove('dragover');
        }, false);
    });
    
    uploadArea.addEventListener('drop', (e) => {
        let dt = e.dataTransfer;
        let files = dt.files;
        handleFiles(files);
    });
    
    videoUpload.addEventListener('change', function() {
        handleFiles(this.files);
    });
    
    function handleFiles(files) {
        if (!files || files.length === 0) return;
        const file = files[0];
        
        if (!file.type.match('video.*')) {
            showError("Please upload a valid video file.");
            return;
        }
        
        uploadVideo(file);
    }
    
    async function uploadVideo(file) {
        // UI Transit
        uploadArea.classList.add('hidden');
        resultsSection.classList.add('hidden');
        loadingSection.classList.remove('hidden');
        
        const formData = new FormData();
        formData.append('video', file);
        
        try {
            const response = await fetch('/api/analyze', {
                method: 'POST',
                body: formData
            });
            
            const data = await response.json();
            
            if (!response.ok || !data.success) {
                throw new Error(data.error || "Analysis failed on the server.");
            }
            
            displayResults(data);
            
        } catch (error) {
            console.error(error);
            showError(error.message);
            // Revert state
            loadingSection.classList.add('hidden');
            uploadArea.classList.remove('hidden');
        }
    }
    
    function displayResults(data) {
        loadingSection.classList.add('hidden');
        resultsSection.classList.remove('hidden');
        
        // Count animation
        const countSpan = document.getElementById('pothole-count');
        animateValue(countSpan, 0, data.total_potholes, 1500);
        
        // Render gallery
        const grid = document.getElementById('gallery-grid');
        grid.innerHTML = '';
        
        if (data.instances && data.instances.length > 0) {
            data.instances.forEach(item => {
                const card = document.createElement('div');
                card.className = 'crop-card';
                card.innerHTML = `
                    <img src="${item.crop_url}" class="crop-img" alt="Pothole Snapshot" onerror="this.src='data:image/svg+xml;utf8,<svg xmlns=\\'http://www.w3.org/2000/svg\\' fill=\\'none\\' viewBox=\\'0 0 24 24\\' stroke=\\'%23555\\'><path stroke-linecap=\\'round\\' stroke-linejoin=\\'round\\' stroke-width=\\'2\\' d=\\'M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z\\'/></svg>'">
                    <div class="crop-info">
                        <span>#${item.id}</span>
                        <span>@ ${Number(item.first_seconds).toFixed(1)}s</span>
                        <span class="crop-conf">${Number(item.confidence * 100).toFixed(0)}%</span>
                    </div>
                `;
                grid.appendChild(card);
            });
        } else {
            grid.innerHTML = '<p style="grid-column: 1/-1; text-align: center; color: var(--text-secondary);">No potholes detected in this video.</p>';
        }
    }
    
    resetBtn.addEventListener('click', () => {
        resultsSection.classList.add('hidden');
        uploadArea.classList.remove('hidden');
        videoUpload.value = '';
    });
    
    function animateValue(obj, start, end, duration) {
        let startTimestamp = null;
        const step = (timestamp) => {
            if (!startTimestamp) startTimestamp = timestamp;
            const progress = Math.min((timestamp - startTimestamp) / duration, 1);
            obj.innerHTML = Math.floor(progress * (end - start) + start);
            if (progress < 1) {
                window.requestAnimationFrame(step);
            }
        };
        window.requestAnimationFrame(step);
    }
    
    function showError(msg) {
        const toast = document.getElementById('error-toast');
        document.getElementById('error-msg').innerText = msg;
        toast.classList.remove('hidden');
        setTimeout(() => {
            toast.classList.add('hidden');
        }, 5000);
    }
});
