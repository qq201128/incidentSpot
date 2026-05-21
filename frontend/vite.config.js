import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const devApiTarget = process.env.VITE_DEV_API_PROXY_TARGET || "http://127.0.0.1:8000";
const devWsTarget = process.env.VITE_DEV_WS_PROXY_TARGET || "ws://127.0.0.1:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    host: true,
    port: 5173,
    // 浏览器只访问 :5173；API/WebSocket 由开发服务器转到本机后端，避免外网需放行 8000 与跨域。
    proxy: {
      "/api": {
        target: devApiTarget,
        changeOrigin: true,
      },
      "/ws": {
        target: devWsTarget,
        ws: true,
        changeOrigin: true,
      },
    },
  },
});
