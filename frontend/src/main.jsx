import React from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import App from "./App";
import FactorLearningPage from "./pages/FactorLearningPage";
import FactorsPage from "./pages/FactorsPage";
import "./app.css";

function Root() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<App />} />
        <Route path="/factors" element={<FactorsPage />} />
        <Route path="/learning" element={<FactorLearningPage />} />
      </Routes>
    </BrowserRouter>
  );
}

createRoot(document.getElementById("app")).render(
  <React.StrictMode>
    <Root />
  </React.StrictMode>
);
