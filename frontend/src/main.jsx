import React from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import App from "./App";
import "./app.css";

const FactorLearningPage = React.lazy(() => import("./pages/FactorLearningPage"));
const FactorsPage = React.lazy(() => import("./pages/FactorsPage"));

function Root() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<App />} />
        <Route path="/factors" element={lazyRoute(<FactorsPage />)} />
        <Route path="/learning" element={lazyRoute(<FactorLearningPage />)} />
      </Routes>
    </BrowserRouter>
  );
}

function lazyRoute(page) {
  return <React.Suspense fallback={<RouteLoading />}>{page}</React.Suspense>;
}

function RouteLoading() {
  return <main className="page-loading" role="status">正在加载页面…</main>;
}

createRoot(document.getElementById("app")).render(
  <React.StrictMode>
    <Root />
  </React.StrictMode>
);
