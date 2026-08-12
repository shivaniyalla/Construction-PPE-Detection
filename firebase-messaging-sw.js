importScripts(
  "https://www.gstatic.com/firebasejs/10.13.2/firebase-app-compat.js"
);

importScripts(
  "https://www.gstatic.com/firebasejs/10.13.2/firebase-messaging-compat.js"
);

firebase.initializeApp({
  apiKey: "AIzaSyDB0kB4le-PFo_AILSxTZG1m7yUsOizrDI",
  authDomain: "guardx-ai.firebaseapp.com",
  projectId: "guardx-ai",
  storageBucket: "guardx-ai.firebasestorage.app",
  messagingSenderId: "551369972933",
  appId: "1:551369972933:web:f4e4186a3501ee0ee8e4ac"
});

const messaging = firebase.messaging();

messaging.onBackgroundMessage(function(payload) {

  console.log(
    "[firebase-messaging-sw.js] Background message received:",
    payload
  );

  const notificationTitle =
    payload.notification?.title || "🚨 GuardX-AI Alert";

  const notificationOptions = {
    body:
      payload.notification?.body ||
      "PPE violation detected. Please take immediate action.",
    icon: "/favicon.ico",
    badge: "/favicon.ico"
  };

  self.registration.showNotification(
    notificationTitle,
    notificationOptions
  );
});
