import { defineConfig } from 'vite';
import path from 'path';


export default defineConfig({
  build: {
    rollupOptions: {
      input: {
        main: path.resolve(__dirname, 'index.html'),
        chat: path.resolve(__dirname, 'chat.html'),
        paymentPlans: path.resolve(__dirname, 'payment-plans.html'),
        resetPassword: path.resolve(__dirname, 'reset-password.html'),
        setNewPassword: path.resolve(__dirname, 'set-new-password.html'),
      },
    },
  },
});
