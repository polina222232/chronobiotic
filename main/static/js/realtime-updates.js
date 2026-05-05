/**
 * Real-time Updates - Notifications only (WebSocket disabled)
 */

class RealtimeUpdates {
    constructor() {
        this.notificationsEnabled = false;
        this.init();
    }

    init() {
        this.requestNotificationPermission();
        // WebSocket полностью отключен - сервер не настроен
        // this.initWebSocket();
    }

    requestNotificationPermission() {
        if ('Notification' in window && Notification.permission === 'default') {
            Notification.requestPermission().then(permission => {
                this.notificationsEnabled = permission === 'granted';
            });
        } else if ('Notification' in window && Notification.permission === 'granted') {
            this.notificationsEnabled = true;
        }
    }

    sendNotification(title, body) {
        if (this.notificationsEnabled && document.hidden) {
            new Notification(title, {
                body: body,
                icon: '/static/image/icon.png',
                silent: false
            });
        }
    }

    notifyNewMessage(role, preview) {
        if (role === 'assistant') {
            this.sendNotification('New AI Response', preview.substring(0, 100));
        }
    }

    showToast(message, type = 'info') {
        const toast = document.createElement('div');
        toast.className = `realtime-toast toast-${type}`;
        toast.textContent = message;

        const colors = {
            info: '#667eea',
            success: '#28a745',
            error: '#dc3545',
            warning: '#ffc107'
        };

        toast.style.cssText = `
            position: fixed;
            top: 80px;
            right: 20px;
            background: ${colors[type] || colors.info};
            color: white;
            padding: 8px 16px;
            border-radius: 8px;
            z-index: 10000;
            font-size: 13px;
            animation: slideInRight 0.3s ease, fadeOut 3s ease 2.7s;
        `;

        document.body.appendChild(toast);
        setTimeout(() => toast.remove(), 3000);
    }
}

document.addEventListener('DOMContentLoaded', () => {
    window.realtimeUpdates = new RealtimeUpdates();
});