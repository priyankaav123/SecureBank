import { useNavigate } from 'react-router-dom';
import { useEffect } from 'react';

const API_BASE_URL = 'http://localhost:5000/api';

interface AnomalyResult {
  anomaly_detected: boolean;
  anomaly_confidence?: number;
  anomaly_reasons?: string[];
  recent_events?: Array<{
    action: string;
    details: string;
  }>;
  reason?: string;
  error?: string;
}

export class AnomalyDetectionService {
  private static instance: AnomalyDetectionService;
  private checkInterval: NodeJS.Timeout | null = null;
  private isChecking = false;

  private constructor() {}

  public static getInstance(): AnomalyDetectionService {
    if (!AnomalyDetectionService.instance) {
      AnomalyDetectionService.instance = new AnomalyDetectionService();
    }
    return AnomalyDetectionService.instance;
  }

  public async checkForAnomaly(): Promise<AnomalyResult> {
    try {
      console.log('🔍 Checking for anomalies...');
      const token = localStorage.getItem('authToken');
      if (!token) {
        console.log('❌ No auth token found');
        return { anomaly_detected: false, reason: 'No auth token' };
      }

      const response = await fetch(`${API_BASE_URL}/behavior/check-anomaly`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        }
      });

      if (!response.ok) {
        console.log(`❌ Anomaly check failed with HTTP ${response.status}`);
        throw new Error(`HTTP ${response.status}`);
      }

      const result: AnomalyResult = await response.json();
      console.log('📊 Anomaly check result:', result);
      
      if (result.anomaly_detected) {
        console.log('🚨 ANOMALY DETECTED ON CLIENT!', result);
      }
      
      return result;
    } catch (error) {
      console.error('❌ Error checking for anomaly:', error);
      return {
        anomaly_detected: false,
        error: error instanceof Error ? error.message : 'Unknown error'
      };
    }
  }

  public handleAnomalyDetected(anomalyData: AnomalyResult, navigate: any) {
    // Store anomaly data for the verification page
    sessionStorage.setItem('anomalyData', JSON.stringify(anomalyData));
    
    // Store the auth token temporarily for AI question generation
    const authToken = localStorage.getItem('authToken');
    if (authToken) {
      sessionStorage.setItem('tempAuthToken', authToken);
    }
    
    // Log the user out
    localStorage.removeItem('authToken');
    
    // Redirect to anomaly verification page
    navigate('/anomaly-verification');
  }

  public startPeriodicChecking(navigate: any, intervalMs: number = 30000) {
    // Periodic checking is now disabled - we only check after transactions
    console.log('Periodic anomaly checking is disabled. Using transaction-based checking.');
  }

  public stopPeriodicChecking() {
    if (this.checkInterval) {
      clearInterval(this.checkInterval);
      this.checkInterval = null;
    }
  }

  public async trackEventAndCheckAnomaly(
    eventType: string, 
    pageUrl?: string, 
    transactionAmount?: number,
    navigate?: any
  ): Promise<boolean> {
    // Check for ANY financial transaction - comprehensive coverage for money loss prevention
    const financialEvents = [
      'bill_payment', 
      'recharge', 
      'own_account_transfer', 
      'beneficiary_transfer',
      'stock_buy',
      'stock_sell',
      'fd_created',
      'tax_payment',
      'transfer',
      'payment'
    ];

    const isFinancialTransaction = financialEvents.includes(eventType) || 
                                  (transactionAmount && transactionAmount > 0);

    if (!isFinancialTransaction) {
      console.log(`Skipping anomaly check for non-financial event: ${eventType}`);
      return false;
    }

    console.log(`🔍 Checking for anomalies after financial transaction: ${eventType}, amount: ${transactionAmount}`);
    
    try {
      // Wait a moment for the backend to process and detect the anomaly
      await new Promise(resolve => setTimeout(resolve, 2000));
      
      const result = await this.checkForAnomaly();
      console.log(`📊 Post-transaction anomaly result:`, result);
      
      if (result.anomaly_detected && navigate) {
        console.log('🚨 Anomaly detected after financial transaction, redirecting to verification');
        this.handleAnomalyDetected(result, navigate);
        return true; // Anomaly detected
      }
      return false; // No anomaly
    } catch (error) {
      console.error('Error in trackEventAndCheckAnomaly:', error);
      return false;
    }
  }
}

// React hook for automatic anomaly detection
export function useAnomalyDetection(enabled: boolean = true) {
  const navigate = useNavigate();
  const anomalyService = AnomalyDetectionService.getInstance();

  useEffect(() => {
    if (!enabled) return;

    // Start periodic checking when component mounts
    anomalyService.startPeriodicChecking(navigate);

    // Cleanup on unmount
    return () => {
      anomalyService.stopPeriodicChecking();
    };
  }, [enabled, navigate]);

  return {
    checkForAnomaly: () => anomalyService.checkForAnomaly(),
    trackEventAndCheckAnomaly: (eventType: string, pageUrl?: string, transactionAmount?: number) =>
      anomalyService.trackEventAndCheckAnomaly(eventType, pageUrl, transactionAmount, navigate)
  };
}

// Helper function to trigger anomaly check after financial transactions
export async function checkAnomalyAfterTransaction(
  transactionType: string,
  amount: number,
  navigate?: any,
  pageUrl?: string
): Promise<boolean> {
  const anomalyService = AnomalyDetectionService.getInstance();
  
  console.log(`🔍 Post-transaction anomaly check: ${transactionType} for $${amount}`);
  
  // Wait for backend to process the transaction and update anomaly detection
  await new Promise(resolve => setTimeout(resolve, 2000));
  
  const result = await anomalyService.checkForAnomaly();
  
  if (result.anomaly_detected && navigate) {
    console.log('🚨 Anomaly detected - redirecting to verification');
    anomalyService.handleAnomalyDetected(result, navigate);
    return true;
  }
  
  console.log('✅ No anomaly detected after transaction');
  return false;
}

// Helper function to trigger anomaly check after important actions (legacy support)
export async function checkAnomalyAfterAction(
  actionType: string,
  navigate?: any,
  transactionAmount?: number
): Promise<boolean> {
  // Use the new transaction-specific function if amount is provided
  if (transactionAmount && transactionAmount > 0) {
    return checkAnomalyAfterTransaction(actionType, transactionAmount, navigate);
  }
  
  const anomalyService = AnomalyDetectionService.getInstance();
  
  // Wait a moment for the backend to process the event
  await new Promise(resolve => setTimeout(resolve, 1000));
  
  const result = await anomalyService.checkForAnomaly();
  
  if (result.anomaly_detected && navigate) {
    anomalyService.handleAnomalyDetected(result, navigate);
    return true;
  }
  
  return false;
}