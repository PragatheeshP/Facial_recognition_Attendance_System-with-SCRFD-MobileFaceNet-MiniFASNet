// Main JavaScript for Classroom Attendance
// Shared utilities across pages

// CSRF token helper (if needed later)
function getCookie(name) {
    const value = `; ${document.cookie}`;
    const parts = value.split(`; ${name}=`);
    if (parts.length === 2) return parts.pop().split(';').shift();
}

// Debounce utility
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// Format date nicely
function formatDate(dateString) {
    if (!dateString) return 'N/A';
    const date = new Date(dateString);
    return date.toLocaleString();
}

// Show toast notification (Bootstrap)
function showToast(message, type = 'info') {
    const toastContainer = document.getElementById('toast-container') || createToastContainer();
    const toast = document.createElement('div');
    toast.className = `toast align-items-center text-white bg-${type} border-0`;
    toast.setAttribute('role', 'alert');
    toast.innerHTML = `
        <div class="d-flex">
            <div class="toast-body">${message}</div>
            <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
        </div>
    `;
    toastContainer.appendChild(toast);
    const bsToast = new bootstrap.Toast(toast, { delay: 5000 });
    bsToast.show();
    toast.addEventListener('hidden.bs.toast', () => toast.remove());
}

function createToastContainer() {
    const container = document.createElement('div');
    container.id = 'toast-container';
    container.className = 'toast-container position-fixed top-0 end-0 p-3';
    container.style.zIndex = '9999';
    document.body.appendChild(container);
    return container;
}

// Check if browser supports required APIs
function checkBrowserSupport() {
    const requirements = [];

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        requirements.push('Camera access (getUserMedia)');
    }

    if (!window.WebSocket) {
        requirements.push('WebSocket (for real-time updates)');
    }

    if (requirements.length > 0) {
        console.warn('Browser missing features:', requirements);
        return {
            supported: false,
            missing: requirements
        };
    }

    return { supported: true };
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    const support = checkBrowserSupport();
    if (!support.supported) {
        console.warn('Some features may not work. Missing:', support.missing);
    }
});
