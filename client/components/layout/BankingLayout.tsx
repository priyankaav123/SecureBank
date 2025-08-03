import { ReactNode, useState } from "react";
import { Link, useLocation, useNavigate } from "react-router-dom";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { useAuth } from "@/contexts/AuthContext";
import { NotificationBell } from "@/components/NotificationBell";
import { useAnomalyDetection } from "@/services/anomalyService";
import {
  User,
  Home,
  CreditCard,
  Send,
  Receipt,
  FileText,
  Settings,
  ChevronDown,
  LogOut,
  Menu,
  X,
  Bell,
  Search,
  Banknote,
  TrendingUp,
  PiggyBank,
} from "lucide-react";

interface BankingLayoutProps {
  children: ReactNode;
}

const navigation = [
  { name: "Dashboard", href: "/", icon: Home },
  { name: "Fixed Deposits", href: "/fixed-deposits", icon: PiggyBank },
  { name: "Stock Investments", href: "/investments", icon: TrendingUp },
  { name: "Account Statement", href: "/statements", icon: FileText },
  { name: "Payments/Transfers", href: "/transfers", icon: Send },
  { name: "Bill Payments", href: "/bills", icon: Receipt },
  { name: "e-Tax", href: "/etax", icon: FileText },
];

export function BankingLayout({ children }: BankingLayoutProps) {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const location = useLocation();
  const navigate = useNavigate();
  const { user, logout } = useAuth();
  
  // Enable automatic anomaly detection for all pages using this layout
  useAnomalyDetection(true);

  const handleLogout = () => {
    logout();
    navigate("/login");
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-banking-gradient shadow-sm border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            {/* Logo and Navigation Toggle */}
            <div className="flex items-center">
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setSidebarOpen(!sidebarOpen)}
                className="lg:hidden text-white hover:bg-white/10"
              >
                <Menu className="h-6 w-6" />
              </Button>
              <div className="flex-shrink-0 flex items-center ml-4 lg:ml-0">
                <div className="bg-white rounded-lg p-2 mr-3">
                  <CreditCard className="h-8 w-8 text-banking-blue-600" />
                </div>
                <div className="text-white">
                  <h1 className="text-xl font-bold">SecureBank</h1>
                  <p className="text-xs text-blue-100">Online Banking</p>
                </div>
              </div>
            </div>

            {/* Search and User Actions */}
            <div className="flex items-center space-x-4">
              <div className="hidden md:block relative">
                <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                  <Search className="h-5 w-5 text-gray-400" />
                </div>
                <input
                  type="text"
                  placeholder="Search..."
                  className="block w-full pl-10 pr-3 py-2 border border-gray-300 rounded-md leading-5 bg-white placeholder-gray-500 focus:outline-none focus:placeholder-gray-400 focus:ring-1 focus:ring-blue-500 focus:border-blue-500 sm:text-sm"
                />
              </div>

              <NotificationBell />

              <div className="flex items-center text-white">
                <User className="h-8 w-8 bg-white/20 rounded-full p-1 mr-2" />
                <div className="hidden sm:block">
                  <p className="text-sm font-medium">
                    {user
                      ? `${user.firstName.toUpperCase()} ${user.lastName.toUpperCase()}`
                      : "USER"}
                  </p>
                  <p className="text-xs text-blue-100">
                    ID:{" "}
                    {user ? `***${user.accountNumber.slice(-4)}` : "***1234"}
                  </p>
                </div>
              </div>

              <Button
                variant="ghost"
                size="sm"
                className="text-white hover:bg-white/10"
                onClick={handleLogout}
              >
                <LogOut className="h-5 w-5" />
              </Button>
            </div>
          </div>
        </div>
      </header>

      <div className="flex">
        {/* Sidebar */}
        <div
          className={cn(
            "fixed inset-y-0 left-0 z-50 w-64 bg-white shadow-lg transform transition-transform duration-300 ease-in-out lg:translate-x-0 lg:static lg:inset-0",
            sidebarOpen ? "translate-x-0" : "-translate-x-full",
          )}
        >
          <div className="flex flex-col h-full">
            {/* Sidebar Header */}
            <div className="flex items-center justify-between h-16 px-6 border-b border-gray-200 lg:hidden">
              <span className="text-lg font-semibold text-gray-900">Menu</span>
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setSidebarOpen(false)}
              >
                <X className="h-6 w-6" />
              </Button>
            </div>

            {/* Quick Actions */}
            <div className="p-6 border-b border-gray-200">
              <h3 className="text-sm font-medium text-gray-900 mb-3">
                Quick Links
              </h3>
              <div className="grid grid-cols-2 gap-2">
                <Button 
                  variant="outline" 
                  size="sm" 
                  className="text-xs"
                  onClick={() => {
                    navigate('/transfers');
                    setSidebarOpen(false);
                  }}
                >
                  Fund Transfer
                </Button>
                <Button 
                  variant="outline" 
                  size="sm" 
                  className="text-xs"
                  onClick={() => {
                    navigate('/fixed-deposits');
                    setSidebarOpen(false);
                  }}
                >
                  New FD
                </Button>
                <Button 
                  variant="outline" 
                  size="sm" 
                  className="text-xs"
                  onClick={() => {
                    navigate('/investments');
                    setSidebarOpen(false);
                  }}
                >
                  Buy Stocks
                </Button>
                <Button 
                  variant="outline" 
                  size="sm" 
                  className="text-xs"
                  onClick={() => {
                    navigate('/');
                    setSidebarOpen(false);
                  }}
                >
                  Portfolio
                </Button>
              </div>
            </div>

            {/* Navigation */}
            <nav className="flex-1 px-6 py-4">
              <ul className="space-y-2">
                {navigation.map((item) => {
                  const Icon = item.icon;
                  const isActive = location.pathname === item.href;

                  return (
                    <li key={item.name}>
                      <Link
                        to={item.href}
                        className={cn(
                          "flex items-center px-3 py-2 text-sm font-medium rounded-md transition-colors",
                          isActive
                            ? "bg-banking-blue-100 text-banking-blue-700 border-r-2 border-banking-blue-600"
                            : "text-gray-600 hover:bg-gray-100 hover:text-gray-900",
                        )}
                        onClick={() => setSidebarOpen(false)}
                      >
                        <Icon
                          className={cn(
                            "mr-3 h-5 w-5",
                            isActive
                              ? "text-banking-blue-600"
                              : "text-gray-400",
                          )}
                        />
                        {item.name}
                      </Link>
                    </li>
                  );
                })}
              </ul>
            </nav>

            {/* Bottom Actions */}
            <div className="p-6 border-t border-gray-200">
              <Button variant="outline" className="w-full mb-2" size="sm">
                <Settings className="mr-2 h-4 w-4" />
                Settings
              </Button>
              <Button
                variant="outline"
                className="w-full"
                size="sm"
                onClick={handleLogout}
              >
                <LogOut className="mr-2 h-4 w-4" />
                Logout
              </Button>
            </div>
          </div>
        </div>

        {/* Overlay */}
        {sidebarOpen && (
          <div
            className="fixed inset-0 z-40 bg-gray-600 bg-opacity-75 lg:hidden"
            onClick={() => setSidebarOpen(false)}
          />
        )}

        {/* Main Content */}
        <main className="flex-1 min-w-0">
          <div className="py-6">
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
              {children}
            </div>
          </div>
        </main>
      </div>
    </div>
  );
}
