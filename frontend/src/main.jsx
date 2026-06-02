import React from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import App from "./App";
import "./app.css";

function Root() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<App />} />
        <Route path="/rule-hit-rate" element={<App />} />
        <Route path="/event-governance" element={<App />} />
        <Route path="/research-dashboard" element={<App />} />
        <Route path="/live-trading" element={<App />} />
        <Route path="/factors" element={<App />} />
        <Route path="/learning" element={<App />} />
      </Routes>
    </BrowserRouter>
  );
}

createRoot(document.getElementById("app")).render(
  <React.StrictMode>
    <Root />
  </React.StrictMode>
);
