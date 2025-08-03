import { useState, useEffect } from 'react';
import { BankingLayout } from "@/components/layout/BankingLayout";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from '@/components/ui/button';
import { Loader2, AlertTriangle, ShieldCheck, Activity, BarChart, CheckCircle, ShieldAlert } from "lucide-react";

// --- Type Definitions ---
interface SessionEvent {
  id: string;
  event_type: string;
  time: string;
  page_url?: string;
  transaction_amount?: number;
  additional_data?: string;
}

// Type for the analysis result from the backend
interface AnalysisResult {
  status: 'Normal' | 'Anomaly' | 'NotApplicable' | 'Error';
  anomaly_confidence_percent?: number;
  raw_score?: number;
  reason?: string;
  anomaly_reasons?: string[];
  recent_events?: Array<{
    action: string;
    details: string;
  }>;
}

const API_BASE_URL = 'http://localhost:5000/api';

export default function SessionMonitor() {
  const [sessionEvents, setSessionEvents] = useState<SessionEvent[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // --- NEW STATE FOR ANALYSIS ---
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null);
  const [analysisError, setAnalysisError] = useState<string | null>(null);
  
  // This useEffect fetches the session data on page load
  useEffect(() => {
    const fetchCurrentSession = async () => {
      // ... (this function remains exactly the same as the previous version)
      setIsLoading(true);
      setError(null);
      try {
        const token = localStorage.getItem('authToken');
        if (!token) throw new Error('You are not logged in.');
        const response = await fetch(`${API_BASE_URL}/behavior/current-session`, {
          method: 'GET',
          headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${token}`},
        });
        const data = await response.json();
        if (!response.ok || !data.success) throw new Error(data.error || `Failed to fetch data.`);
        setSessionEvents(data.session_events || []);
      } catch (err: any) {
        setError(err.message);
      } finally {
        setIsLoading(false);
      }
    };
    fetchCurrentSession();
  }, []);

  // --- NEW FUNCTION TO HANDLE ANALYSIS ---
  const handleAnalyzeSession = async () => {
    if (sessionEvents.length === 0) return;

    setIsAnalyzing(true);
    setAnalysisError(null);
    setAnalysisResult(null);

    try {
      const token = localStorage.getItem('authToken');
      if (!token) throw new Error('You are not logged in.');

      const response = await fetch(`${API_BASE_URL}/behavior/analyze-session`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${token}` },
        body: JSON.stringify({ events: sessionEvents }), // Send the events we fetched
      });

      const resultData: AnalysisResult = await response.json();

      if (!response.ok) {
        throw new Error((resultData as any).error || 'Analysis failed.');
      }
      
      setAnalysisResult(resultData);

    } catch (err: any) {
      setAnalysisError(err.message || 'An unknown error occurred during analysis.');
    } finally {
      setIsAnalyzing(false);
    }
  };

  const getResultCardClass = () => {
    if (!analysisResult) return 'border-gray-200';
    switch (analysisResult.status) {
      case 'Anomaly': return 'border-red-500 bg-red-50';
      case 'Normal': return 'border-green-500 bg-green-50';
      default: return 'border-yellow-500 bg-yellow-50';
    }
  };

  const formatTime = (isoString: string) => new Date(isoString).toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: true });

  return (
    <BankingLayout>
      <div className="space-y-6 animate-slide-in">
        <div className="flex justify-between items-center">
          <div>
            <h1 className="text-3xl font-bold text-gray-800">Current Session Activity Monitor</h1>
            <p className="text-gray-600">A live log of all actions performed since you logged in.</p>
          </div>
          {/* --- THE NEW ANALYSIS BUTTON --- */}
          <Button 
            onClick={handleAnalyzeSession}
            disabled={isAnalyzing || isLoading || sessionEvents.length === 0}
            size="lg"
          >
            {isAnalyzing ? (
              <Loader2 className="mr-2 h-5 w-5 animate-spin" />
            ) : (
              <BarChart className="mr-2 h-5 w-5" />
            )}
            Analyze Session
          </Button>
        </div>

        {/* --- ANALYSIS RESULT DISPLAY --- */}
        {analysisError && <p className="text-red-600 text-center p-2">{analysisError}</p>}
        {analysisResult && (
          <Card className={`border-2 ${getResultCardClass()}`}>
            <CardHeader>
              <CardTitle className="flex items-center text-xl">
                {analysisResult.status === 'Normal' && <CheckCircle className="mr-3 h-7 w-7 text-green-600" />}
                {analysisResult.status === 'Anomaly' && <ShieldAlert className="mr-3 h-7 w-7 text-red-600" />}
                {analysisResult.status === 'NotApplicable' && <AlertTriangle className="mr-3 h-7 w-7 text-yellow-600" />}
                Analysis Result: {analysisResult.status}
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              {analysisResult.anomaly_confidence_percent !== undefined && (
                <p><strong>Anomaly Confidence:</strong> {analysisResult.anomaly_confidence_percent.toFixed(2)}%</p>
              )}
              {analysisResult.reason && (
                <p><strong>Reason:</strong> {analysisResult.reason}</p>
              )}
              
              {/* Display anomaly reasons if available */}
              {analysisResult.anomaly_reasons && analysisResult.anomaly_reasons.length > 0 && (
                <div className="space-y-2">
                  <p className="font-semibold text-red-700">Anomaly Reasons:</p>
                  <ul className="list-disc list-inside space-y-1 text-sm bg-red-50 p-3 rounded-md border border-red-200">
                    {analysisResult.anomaly_reasons.map((reason, index) => (
                      <li key={index} className="text-red-800">{reason}</li>
                    ))}
                  </ul>
                </div>
              )}
              
              {/* Display recent events if available */}
              {analysisResult.recent_events && analysisResult.recent_events.length > 0 && (
                <div className="space-y-2">
                  <p className="font-semibold text-gray-700">Recent Session Activity:</p>
                  <div className="bg-gray-50 p-3 rounded-md border border-gray-200 max-h-48 overflow-y-auto">
                    <ul className="space-y-2 text-sm">
                      {analysisResult.recent_events.map((event, index) => (
                        <li key={index} className="flex justify-between items-start">
                          <span className="font-medium text-gray-800 capitalize">
                            {event.action.replace(/_/g, ' ')}
                          </span>
                          <span className="text-gray-600 text-xs ml-2">
                            {event.details}
                          </span>
                        </li>
                      ))}
                    </ul>
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        )}

        {/* --- SESSION DATA DISPLAY --- */}
        {isLoading ? (
          <div className="flex items-center justify-center h-48">
            <Loader2 className="h-8 w-8 animate-spin text-blue-600" />
            <span className="ml-3 text-lg">Loading your current session activity...</span>
          </div>
        ) : error ? (
          <div className="flex flex-col items-center justify-center h-48 text-red-600">
            <AlertTriangle className="h-10 w-10 mb-2" />
            <p className="font-semibold">{error}</p>
          </div>
        ) : sessionEvents.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-48 text-gray-500">
            <ShieldCheck className="h-10 w-10 mb-2" />
            <p className="font-semibold">No activity recorded in this session yet.</p>
          </div>
        ) : (
          <Card className="shadow-md">
            <CardHeader>
              <CardTitle className="flex items-center text-xl text-gray-800">
                <Activity className="h-6 w-6 mr-3 text-blue-600" />
                Activity Log ({sessionEvents.length} events)
              </CardTitle>
            </CardHeader>
            <CardContent>
              <ul className="divide-y divide-gray-200">
                {sessionEvents.map((event) => {
                  // Parse additional data to check for biometric failures
                  let additionalData = {};
                  try {
                    additionalData = event.additional_data ? JSON.parse(event.additional_data) : {};
                  } catch (e) {
                    additionalData = {};
                  }
                  
                  // Check if this is a biometric authentication failure
                  const isBiometricFailure = event.event_type === 'login_failed' && 
                    (event.page_url === '/login' || additionalData.reason?.includes('Biometric') || additionalData.reason?.includes('biometric'));
                  
                  return (
                    <li key={event.id} className={`py-3 px-1 grid grid-cols-3 md:grid-cols-4 gap-4 items-center ${
                      isBiometricFailure ? 'bg-red-50 border border-red-200 rounded-md' : ''
                    }`}>
                      <div className="font-mono text-gray-700">{formatTime(event.time)}</div>
                      <div className={`font-semibold col-span-2 md:col-span-1 ${
                        isBiometricFailure ? 'text-red-700' : 'text-gray-900'
                      }`}>
                        {isBiometricFailure && <ShieldAlert className="inline w-4 h-4 mr-1" />}
                        {event.event_type.replace(/_/g, ' ')}
                        {isBiometricFailure && (
                          <span className="text-xs text-red-600 block">
                            🔐 {additionalData.reason || 'Biometric authentication failed'}
                          </span>
                        )}
                      </div>
                      <div className="hidden md:block text-gray-600">{event.page_url}</div>
                      <div className="text-right font-medium">
                        {event.transaction_amount && event.transaction_amount > 0 ? (
                          <span className="text-green-600">₹{event.transaction_amount.toLocaleString()}</span>
                        ) : isBiometricFailure ? (
                          <span className="text-red-600 text-xs">SECURITY ALERT</span>
                        ) : null}
                      </div>
                    </li>
                  );
                })}
              </ul>
            </CardContent>
          </Card>
        )}
      </div>
    </BankingLayout>
  );
}