import { useState } from 'react';
import { BankingLayout } from "@/components/layout/BankingLayout";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from '@/components/ui/button';
import { AlertTriangle, Activity, Bug } from "lucide-react";

const API_BASE_URL = 'http://localhost:5000/api';

export default function AnomalyDebug() {
  const [debugResult, setDebugResult] = useState<any>(null);
  const [isChecking, setIsChecking] = useState(false);
  const [anomalyResult, setAnomalyResult] = useState<any>(null);

  const handleDebugCheck = async () => {
    setIsChecking(true);
    try {
      const token = localStorage.getItem('authToken');
      const response = await fetch(`${API_BASE_URL}/debug/force-anomaly-check`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        }
      });

      const result = await response.json();
      setDebugResult(result);
      console.log('Debug result:', result);
    } catch (error) {
      console.error('Debug check failed:', error);
      setDebugResult({ error: error.message });
    } finally {
      setIsChecking(false);
    }
  };

  const handleRegularCheck = async () => {
    setIsChecking(true);
    try {
      const token = localStorage.getItem('authToken');
      const response = await fetch(`${API_BASE_URL}/behavior/check-anomaly`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        }
      });

      const result = await response.json();
      setAnomalyResult(result);
      console.log('Regular anomaly check result:', result);
      
      // If anomaly detected, trigger the redirect flow
      if (result.anomaly_detected) {
        console.log('🚨 Anomaly detected! Triggering redirect...');
        sessionStorage.setItem('anomalyData', JSON.stringify(result));
        localStorage.removeItem('authToken');
        window.location.href = '/anomaly-verification';
      }
    } catch (error) {
      console.error('Regular check failed:', error);
      setAnomalyResult({ error: error.message });
    } finally {
      setIsChecking(false);
    }
  };

  return (
    <BankingLayout>
      <div className="space-y-6">
        <div>
          <h1 className="text-3xl font-bold text-gray-800">Anomaly Detection Debug</h1>
          <p className="text-gray-600">Test and debug the anomaly detection system</p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center">
                <Bug className="mr-2 h-6 w-6" />
                Debug Anomaly Check
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <p className="text-sm text-gray-600">
                Runs a detailed debug check to see what's happening with anomaly detection
              </p>
              <Button onClick={handleDebugCheck} disabled={isChecking} className="w-full">
                {isChecking ? 'Checking...' : 'Run Debug Check'}
              </Button>
              
              {debugResult && (
                <div className="mt-4 p-4 bg-gray-50 rounded-lg">
                  <pre className="text-xs overflow-auto max-h-64">
                    {JSON.stringify(debugResult, null, 2)}
                  </pre>
                </div>
              )}
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="flex items-center">
                <Activity className="mr-2 h-6 w-6" />
                Regular Anomaly Check
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <p className="text-sm text-gray-600">
                Runs the normal anomaly check that happens in the background
              </p>
              <Button onClick={handleRegularCheck} disabled={isChecking} className="w-full">
                {isChecking ? 'Checking...' : 'Run Regular Check'}
              </Button>
              
              {anomalyResult && (
                <div className="mt-4 p-4 bg-gray-50 rounded-lg">
                  <pre className="text-xs overflow-auto max-h-64">
                    {JSON.stringify(anomalyResult, null, 2)}
                  </pre>
                </div>
              )}
            </CardContent>
          </Card>
        </div>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center">
              <AlertTriangle className="mr-2 h-6 w-6" />
              Debug Information
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div>
                <h3 className="font-semibold">Current Status:</h3>
                <ul className="text-sm space-y-1 mt-2">
                  <li>• Check browser console for detailed logging</li>
                  <li>• Debug endpoint will show profiler status and recent events</li>
                  <li>• Regular check mimics the background anomaly detection</li>
                </ul>
              </div>
              
              <div>
                <h3 className="font-semibold">Expected Behavior:</h3>
                <ul className="text-sm space-y-1 mt-2">
                  <li>• If you made large transactions ($1000), anomaly should be detected</li>
                  <li>• If behavioral model isn't trained, checks will return "NotApplicable"</li>
                  <li>• If anomaly is detected, you should be redirected to verification page</li>
                </ul>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </BankingLayout>
  );
}