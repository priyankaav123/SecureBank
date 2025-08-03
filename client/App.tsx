import "./global.css";

import { Toaster } from "@/components/ui/toaster";
import { createRoot } from "react-dom/client";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import { AuthProvider } from "@/contexts/AuthContext";
import { NotificationProvider } from "@/contexts/NotificationContext";
import { ProtectedRoute } from "@/components/ProtectedRoute";
import Index from "./pages/Index";
import NotFound from "./pages/NotFound";
import Login from "./pages/Login";
import Register from "./pages/Register";
import Accounts from "./pages/Accounts";
import AccountStatement from "./pages/AccountStatement";
import PaymentsTransfers from "./pages/PaymentsTransfers";
import BillPayments from "./pages/BillPayments";
import ETax from "./pages/ETax";
import FixedDeposits from "./pages/FixedDeposits";
import StockInvestments from "./pages/StockInvestments";
import SessionMonitor from "./pages/SessionMonitor";
import AnomalyDebug from "./pages/AnomalyDebug";
import AnomalyVerification from "./pages/AnomalyVerification";
const App = () => (
  <TooltipProvider>
    <AuthProvider>
      <NotificationProvider>
        <Toaster />
        <Sonner />
      <BrowserRouter>
        <Routes>
            {/* Public routes */}
            <Route path="/login" element={<Login />} />
            <Route path="/register" element={<Register />} />

            {/* Protected routes */}
            <Route
              path="/"
              element={
                <ProtectedRoute>
                  <Index />
                </ProtectedRoute>
              }
            />
            <Route
              path="/accounts"
              element={
                <ProtectedRoute>
                  <Accounts />
                </ProtectedRoute>
              }
            />
            <Route
              path="/fixed-deposits"
              element={
                <ProtectedRoute>
                  <FixedDeposits />
                </ProtectedRoute>
              }
            />
            <Route
              path="/investments"
              element={
                <ProtectedRoute>
                  <StockInvestments />
                </ProtectedRoute>
              }
            />
            <Route
              path="/statements"
              element={
                <ProtectedRoute>
                  <AccountStatement />
                </ProtectedRoute>
              }
            />
            <Route
              path="/transfers"
              element={
                <ProtectedRoute>
                  <PaymentsTransfers />
                </ProtectedRoute>
              }
            />
            <Route
              path="/bills"
              element={
                <ProtectedRoute>
                  <BillPayments />
                </ProtectedRoute>
              }
            />
            <Route
              path="/etax"
              element={
                <ProtectedRoute>
                  <ETax />
                </ProtectedRoute>
              }
            />
            <Route
              path="/session-monitor"
              element={
                <ProtectedRoute>
                  <SessionMonitor />
                </ProtectedRoute>
              }
            />
            <Route
              path="/anomaly-debug"
              element={
                <ProtectedRoute>
                  <AnomalyDebug />
                </ProtectedRoute>
              }
            />
            <Route
              path="/anomaly-verification"
              element={
                <ProtectedRoute>
                  <AnomalyVerification/>
                </ProtectedRoute>
              }
            />
            {/* Catch-all route */}
            <Route path="*" element={<NotFound />} />
          </Routes>
        </BrowserRouter>
      </NotificationProvider>
    </AuthProvider>
  </TooltipProvider>
);

createRoot(document.getElementById("root")!).render(<App />);
