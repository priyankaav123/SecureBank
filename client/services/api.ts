// API service for frontend-backend communication

const API_BASE = "http://localhost:5000/api";

// Helper function to get auth headers
function getAuthHeaders(): HeadersInit {
  const token = localStorage.getItem("authToken");
  return {
    "Content-Type": "application/json",
    ...(token && { Authorization: `Bearer ${token}` }),
  };
}

// Helper function to handle API responses
async function handleResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const errorData = await response
      .json()
      .catch(() => ({ error: "Network error" }));
    throw new Error(errorData.error || `HTTP error ${response.status}`);
  }
  return response.json();
}

// Stock API
export const stocksApi = {
  getAllStocks: async () => {
    const response = await fetch(`${API_BASE}/stocks`);
    return handleResponse(response);
  },

  getStock: async (symbol: string) => {
    const response = await fetch(`${API_BASE}/stocks/${symbol}`);
    return handleResponse(response);
  },

  buyStock: async (symbol: string, shares: number) => {
    const response = await fetch(`${API_BASE}/stocks/buy`, {
      method: "POST",
      headers: getAuthHeaders(),
      body: JSON.stringify({ symbol, shares }),
    });
    return handleResponse(response);
  },

  sellStock: async (symbol: string, shares: number) => {
    const response = await fetch(`${API_BASE}/stocks/sell`, {
      method: "POST",
      headers: getAuthHeaders(),
      body: JSON.stringify({ symbol, shares }),
    });
    return handleResponse(response);
  },
};

// Portfolio API
export const portfolioApi = {
  getPortfolio: async () => {
    const response = await fetch(`${API_BASE}/portfolio`, {
      headers: getAuthHeaders(),
    });
    return handleResponse(response);
  },
};

// Transactions API
export const transactionsApi = {
  getTransactions: async (type?: string, limit = 20, offset = 0) => {
    const params = new URLSearchParams({
      limit: limit.toString(),
      offset: offset.toString(),
      ...(type && { type }),
    });

    const response = await fetch(`${API_BASE}/transactions?${params}`, {
      headers: getAuthHeaders(),
    });
    return handleResponse(response);
  },
};

// User API
export const userApi = {
  getProfile: async () => {
    const response = await fetch(`${API_BASE}/auth/profile`, {
      headers: getAuthHeaders(),
    });
    return handleResponse(response);
  },
};

// Fixed Deposits API
export const fixedDepositsApi = {
  getFixedDeposits: async () => {
    const response = await fetch(`${API_BASE}/fixed-deposits`, {
      headers: getAuthHeaders(),
    });
    return handleResponse(response);
  },

  createFixedDeposit: async (amount: number, tenure: number, type: string) => {
    const response = await fetch(`${API_BASE}/fixed-deposits`, {
      method: "POST",
      headers: getAuthHeaders(),
      body: JSON.stringify({ amount, tenure, type }),
    });
    return handleResponse(response);
  },

  getFDDetails: async (fdId: string) => {
    const response = await fetch(`${API_BASE}/fixed-deposits/${fdId}/details`, {
      headers: getAuthHeaders(),
    });
    return handleResponse(response);
  },

  getFDCertificate: async (fdId: string) => {
    const response = await fetch(`${API_BASE}/fixed-deposits/${fdId}/certificate`, {
      headers: getAuthHeaders(),
    });
    return handleResponse(response);
  },

  exportFixedDeposits: async () => {
    const response = await fetch(`${API_BASE}/fixed-deposits/export`, {
      headers: getAuthHeaders(),
    });
    return handleResponse(response);
  },
};

// Account Statement API
export const accountStatementApi = {
  getStatement: async (params: {
    start_date?: string;
    end_date?: string;
    period?: string;
    records_per_page?: string;
  }) => {
    const queryParams = new URLSearchParams();
    if (params.start_date) queryParams.append('start_date', params.start_date);
    if (params.end_date) queryParams.append('end_date', params.end_date);
    if (params.period) queryParams.append('period', params.period);
    if (params.records_per_page) queryParams.append('records_per_page', params.records_per_page);

    const response = await fetch(`${API_BASE}/account-statement?${queryParams}`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${localStorage.getItem("authToken")}`,
      },
    });
    return response.json();
  },

  exportStatement: async (data: {
    format: 'pdf' | 'excel';
    start_date?: string;
    end_date?: string;
    period?: string;
  }) => {
    const response = await fetch(`${API_BASE}/account-statement/export`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${localStorage.getItem("authToken")}`,
      },
      body: JSON.stringify(data),
    });
    return response.json();
  },
};

// Types for API responses
export interface Stock {
  symbol: string;
  name: string;
  price: number;
  change: number;
  changePercent: number;
  volume: string;
  high: number;
  low: number;
  category: string;
  lastUpdated: string;
}

export interface Portfolio {
  id: string;
  userId: string;
  symbol: string;
  shares: number;
  buyPrice: number;
  totalInvestment: number;
  purchaseDate: string;
  currentPrice: number;
  currentValue: number;
  gainLoss: number;
  gainLossPercent: number;
  stockData?: Stock;
}

export interface Transaction {
  id: string;
  userId: string;
  type:
    | "BUY"
    | "SELL"
    | "DEPOSIT"
    | "WITHDRAW"
    | "TRANSFER_IN"
    | "TRANSFER_OUT";
  symbol?: string;
  shares?: number;
  amount: number;
  price?: number;
  description: string;
  status: "PENDING" | "COMPLETED" | "FAILED";
  createdAt: string;
  completedAt?: string;
  reference?: string;
  counterparty?: string;
}

export interface PortfolioSummary {
  totalValue: number;
  totalInvestment: number;
  totalGainLoss: number;
  totalGainLossPercent: number;
  holdings: number;
}

export interface UserProfile {
  user: any;
  portfolio: Portfolio[];
  portfolioSummary: PortfolioSummary;
  recentTransactions: Transaction[];
  fixedDeposits: any[];
}

export interface FixedDeposit {
  id: string;
  userId: string;
  amount: number;
  interestRate: number;
  tenure: number;
  startDate: string;
  maturityDate: string;
  type: string;
  status: string;
  interestEarned: number;
  maturityAmount: number;
  createdAt?: string;
}

export interface FDDetails {
  id: string;
  amount: number;
  interestRate: number;
  tenure: number;
  startDate: string;
  maturityDate: string;
  type: string;
  status: string;
  interestEarned: number;
  maturityAmount: number;
  createdAt: string;
  customerName: string;
  accountNumber: string;
}

export interface FDCertificate {
  certificateNumber: string;
  customerName: string;
  accountNumber: string;
  phone: string;
  amount: number;
  interestRate: number;
  tenure: number;
  startDate: string;
  maturityDate: string;
  type: string;
  maturityAmount: number;
  issueDate: string;
  branch: string;
  bankName: string;
}

export interface FDExportData {
  customerName: string;
  accountNumber: string;
  exportDate: string;
  summary: {
    totalInvestment: number;
    totalInterestEarned: number;
    activeFDs: number;
    totalFDs: number;
  };
  fixedDeposits: FixedDeposit[];
}

export interface StatementEntry {
  date: string;
  description: string;
  debit: string;
  credit: string;
  balance: string;
  type: string;
  amount: number;
  status: string;
}

export interface AccountStatement {
  accountNumber: string;
  period: string;
  startDate?: string;
  endDate?: string;
  totalRecords: number;
  entries: StatementEntry[];
}