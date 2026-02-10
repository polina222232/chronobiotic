/**
 * Real-time Updates Module
 * Handles real-time notifications and updates
 */

class RealtimeUpdates {
    constructor() {
        this.notificationsEnabled = false;
        this.init();
    }

    init() {
        this.requestNotificationPermission();
        this.setupEventSource();
    }

    requestNotificationPermission() {
        if ('Notification' in window && Notification.permission === 'default') {
            Notification.requestPermission().then(permission => {
                this.notificationsEnabled = permission === 'granted';
            });
        }
    }

    setupEventSource() {
        // For future server-sent events
        // const eventSource = new EventSource('/api/events/');
        // eventSource.onmessage = (event) => this.handleEvent(event);
    }

    sendNotification(title, body, icon = null) {
        if (this.notificationsEnabled && document.hidden) {
            new Notification(title, {
                body: body,
                icon: icon || '/static/image/icon.png',
                silent: false
            });
        }
    }

    showToast(message, type = 'info') {
        const toast = document.createElement('div');
        toast.className = `realtime-toast toast-${type}`;

        const icons = {
            info: 'fa-info-circle',
            success: 'fa-check-circle',
            error: 'fa-exclamation-circle',
            warning: 'fa-exclamation-triangle'
        };

        toast.innerHTML = `
            <i class="fas ${icons[type]}"></i>
            <span>${message}</span>
        `;

        toast.style.cssText = `
            position: fixed;
            top: 80px;
            right: 20px;
            background: ${type === 'error' ? '#dc3545' : type === 'success' ? '#28a745' : '#667eea'};
            color: white;
            padding: 12px 20px;
            border-radius: 8px;
            z-index: 10000;
            animation: slideInRight 0.3s ease, fadeOut 3s ease 2.7s;
            display: flex;
            gap: 10px;
            align-items: center;
            box-shadow: 0 2px 10px rgba(0,0,0,0.2);
        `;

        document.body.appendChild(toast);
        setTimeout(() => toast.remove(), 3000);
    }

    handleEvent(event) {
        const data = JSON.parse(event.data);
        this.showToast(data.message, data.type);

        if (data.notification) {
            this.sendNotification(data.title, data.message);
        }
    }
}

const realtimeUpdates = new RealtimeUpdates();