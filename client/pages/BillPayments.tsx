import React, { useState, useEffect, useMemo, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Alert, AlertDescription } from '@/components/ui/alert';
import {
  Receipt,
  Zap,
  Smartphone,
  Wifi,
  Building,
  CreditCard,
  Clock,
  ChevronRight,
  Star,
  ArrowLeft,
  CheckCircle,
  AlertCircle,
  Wallet,
  Download,
  Shield,
  Plus,
  Trash2,
  Settings,
  XCircle
} from 'lucide-react';
import { BankingLayout } from '@/components/layout/BankingLayout';
import { useAuth } from '@/contexts/AuthContext';
import { useNotifications } from '@/contexts/NotificationContext';
import { billPaymentsApi, Biller, AutoPayRule } from '@/services/billPaymentsapi';
import { checkAnomalyAfterTransaction } from '@/services/anomalyService';

// --- CONSTANTS (moved outside for better organization) ---
const billCategories = [
    { title: "Pay Bills", description: "Pay utility bills and other services", icon: Receipt, color: "text-blue-600", bgColor: "bg-blue-100", popular: true },
    { title: "Payment History", description: "View your past bill payments", icon: Clock, color: "text-green-600", bgColor: "bg-green-100" },
    { title: "Topup Recharge", description: "Mobile and DTH recharge services", icon: Smartphone, color: "text-orange-600", bgColor: "bg-orange-100", popular: true },
    { title: "Auto Pay Setup", description: "Set up automatic bill payments", icon: Shield, color: "text-purple-600", bgColor: "bg-purple-100" },
];

// Available bill types for new registrations
const availableBillTypes = [
    { category: "Utilities", bills: [
        { id: 'electricity', name: "Electricity Bill", icon: Zap, providers: [ "KSEB", "MSEB", "TSEB", "BESCOM" ] },
        { id: 'gas', name: "Gas Bill", icon: Building, providers: [ "Indane", "Bharat Gas", "HP Gas" ] },
        { id: 'water', name: "Water Bill", icon: Building, providers: [ "Municipal Corporation", "Water Board" ] }
    ]},
    { category: "Telecom", bills: [
        { id: 'mobile', name: "Mobile Recharge", icon: Smartphone, providers: [ "Airtel", "Jio", "BSNL", "Vi" ] },
        { id: 'dth', name: "DTH Recharge", icon: Wifi, providers: [ "Tata Sky", "Airtel Digital TV", "Dish TV" ] },
        { id: 'broadband', name: "Broadband", icon: Wifi, providers: [ "BSNL", "Airtel", "Jio Fiber" ] }
    ]},
    { category: "Financial", bills: [
        { id: 'creditcard', name: "Credit Card", icon: CreditCard, providers: [ "SBI Card", "HDFC", "ICICI", "Axis" ] },
        { id: 'loan', name: "Loan EMI", icon: Building, providers: [ "Home Loan", "Personal Loan", "Car Loan" ] },
        { id: 'insurance', name: "Insurance Premium", icon: Shield, providers: [ "LIC", "SBI Life", "HDFC Life" ] }
    ]}
];
const rechargePlans = {
    prepaid: [ { amount: 199, validity: '28 days', data: '1.5GB/day', description: 'Unlimited calls & SMS' }, { amount: 299, validity: '28 days', data: '2GB/day', description: 'Unlimited calls & SMS' }, { amount: 399, validity: '56 days', data: '2.5GB/day', description: 'Unlimited calls & SMS' }, { amount: 599, validity: '84 days', data: '2GB/day', description: 'Unlimited calls & SMS' }, { amount: 699, validity: '84 days', data: '3GB/day', description: 'Unlimited calls & SMS' }, { amount: 999, validity: '84 days', data: '3GB/day', description: 'Disney+ Hotstar included' } ],
    postpaid: [ { amount: 399, features: 'Unlimited calls, 40GB data', description: '100 SMS/month' }, { amount: 599, features: 'Unlimited calls, 75GB data', description: 'Netflix Basic included' }, { amount: 999, features: 'Unlimited calls, 150GB data', description: 'Netflix Premium + Prime' }, { amount: 1299, features: 'Unlimited calls, 200GB data', description: 'All OTT platforms' } ]
};

// --- HELPER FUNCTION ---
const generateReceipt = (payment) => {
    const receiptContent = `
PAYMENT RECEIPT
===============
Transaction ID: ${payment.id}
Date: ${new Date(payment.createdAt || payment.date).toLocaleDateString('en-IN')}
Time: ${new Date().toLocaleTimeString('en-IN')}
Service: ${payment.description || payment.service}
Provider: ${payment.counterparty || payment.provider}
Customer: ${payment.customerName || 'Customer'}
Amount: ₹${payment.amount.toLocaleString('en-IN')}
Status: ${payment.status}
Thank you for using our service!
===============================
    `;
    const blob = new Blob([receiptContent], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `receipt-${payment.transactionId}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
};

// --- VIEW COMPONENTS (Moved outside main component to prevent re-creation on render) ---

const DashboardView = ({ accountBalance, paymentHistory, handleCategoryClick, setCurrentView, isLoadingHistory }) => (
    <div className="space-y-6">
      <div className="bg-gradient-to-r from-blue-500 to-blue-600 rounded-lg p-6 text-white">
        <div className="flex items-center justify-between">
          <div className="flex items-center">
            <Receipt className="h-8 w-8 mr-4" />
            <div>
              <h1 className="text-2xl font-bold">Bill Payments</h1>
              <p className="text-blue-100">Pay all your bills conveniently in one place</p>
            </div>
          </div>
          <div className="text-right">
            <div className="flex items-center mb-2">
              <Wallet className="h-5 w-5 mr-2" />
              <span className="text-sm">Available Balance</span>
            </div>
            <p className="text-2xl font-bold">₹{accountBalance.toLocaleString('en-IN')}</p>
          </div>
        </div>
      </div>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {billCategories.map((category) => {
          const Icon = category.icon;
          return (
            <Card key={category.title} className="hover:shadow-lg transition-shadow cursor-pointer group relative" onClick={() => handleCategoryClick(category)}>
              {category.popular && (<Badge className="absolute -top-2 -right-2 bg-orange-500 hover:bg-orange-600"><Star className="h-3 w-3 mr-1" />Popular</Badge>)}
              <CardContent className="p-6 text-center">
                <div className={`${category.bgColor} w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-4`}><Icon className={`h-8 w-8 ${category.color}`} /></div>
                <h3 className="font-semibold text-gray-900 mb-2 group-hover:text-blue-600 transition-colors">{category.title}</h3>
                <p className="text-sm text-gray-600 mb-4">{category.description}</p>
                <Button variant="outline" size="sm" className="w-full">Access <ChevronRight className="h-4 w-4 ml-1" /></Button>
              </CardContent>
            </Card>
          );
        })}
      </div>
      <Card>
        <CardHeader><CardTitle className="text-lg font-semibold flex items-center"><Clock className="h-5 w-5 mr-2 text-blue-600" />Recent Payments</CardTitle></CardHeader>
        <CardContent>
          <div className="space-y-4">
            {isLoadingHistory ? (
              <div className="text-center py-8 text-gray-500">
                <div className="animate-pulse">
                  <div className="h-12 w-12 mx-auto mb-4 bg-gray-200 rounded"></div>
                  <div className="h-4 bg-gray-200 rounded w-32 mx-auto mb-2"></div>
                  <div className="h-3 bg-gray-200 rounded w-48 mx-auto"></div>
                </div>
              </div>
            ) : paymentHistory.length === 0 ? (
              <div className="text-center py-8 text-gray-500">
                <Receipt className="h-12 w-12 mx-auto mb-4 text-gray-300" />
                <p>No payments made yet</p>
                <p className="text-sm">Your recent transactions will appear here</p>
              </div>
            ) : (
              <>
                {paymentHistory
                  .filter(payment => 
                    payment.type === 'BILL_PAYMENT' || 
                    payment.type === 'RECHARGE' ||
                    payment.type === 'TAX_DIRECT' ||
                    payment.type === 'TAX_GST' ||
                    payment.type === 'TAX_STATE'
                  )
                  .slice(0, 3)
                  .map((payment) => (
                  <div key={payment.id} className="flex items-center justify-between p-3 border rounded-lg hover:bg-gray-50">
                    <div className="flex items-center space-x-3">
                      <div className={`p-2 rounded-lg ${payment.type === 'BILL_PAYMENT' ? 'bg-blue-100' : payment.type === 'RECHARGE' ? 'bg-orange-100' : 'bg-green-100'}`}>
                        {payment.type === 'BILL_PAYMENT' ? <Receipt className="h-4 w-4 text-blue-600" /> : 
                         payment.type === 'RECHARGE' ? <Smartphone className="h-4 w-4 text-orange-600" /> : 
                         <Receipt className="h-4 w-4 text-green-600" />}
                      </div>
                      <div>
                        <p className="font-medium text-gray-900">{payment.description}</p>
                        <p className="text-sm text-gray-600">{payment.counterparty || payment.reference}</p>
                      </div>
                    </div>
                    <div className="text-right">
                      <p className="font-semibold text-gray-900">₹{payment.amount.toLocaleString("en-IN")}</p>
                      <Badge variant="default" className={`text-xs ${payment.status === 'COMPLETED' ? 'bg-green-100 text-green-800' : 'bg-yellow-100 text-yellow-800'}`}>{payment.status}</Badge>
                    </div>
                  </div>
                ))}
                <Button variant="outline" className="w-full" onClick={() => setCurrentView('history')}>View All Payment History</Button>
              </>
            )}
          </div>
        </CardContent>
      </Card>
    </div>
);
const BillTypesView = ({ resetToDashboard, handleBillSelect, registeredBillers, handleAddBiller, handleDeleteBiller, isLoadingBillers }) => (
    <div className="space-y-6">
      <div className="flex items-center space-x-4">
        <Button variant="ghost" onClick={resetToDashboard}><ArrowLeft className="h-4 w-4 mr-2" />Back to Dashboard</Button>
        <h2 className="text-2xl font-bold">Bill Payments</h2>
      </div>
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card>
          <CardHeader><CardTitle className="text-lg font-semibold">Your Registered Billers</CardTitle></CardHeader>
          <CardContent>
            {isLoadingBillers ? (
              <div className="space-y-3">
                {[1,2,3].map(i => (
                  <div key={i} className="animate-pulse">
                    <div className="h-16 bg-gray-200 rounded-lg"></div>
                  </div>
                ))}
              </div>
            ) : registeredBillers.length === 0 ? (
              <div className="text-center py-8 text-gray-500">
                <Receipt className="h-12 w-12 mx-auto mb-4 text-gray-300" />
                <p>No billers registered yet</p>
                <p className="text-sm">Add billers to pay bills easily</p>
              </div>
            ) : (
              <div className="space-y-3">
                {registeredBillers.map((biller) => (
                  <div key={biller.id} className="border rounded-lg p-4 flex items-center justify-between">
                    <div className="flex items-center space-x-3">
                      <div className="bg-blue-100 p-2 rounded-lg">
                        <Receipt className="h-5 w-5 text-blue-600" />
                      </div>
                      <div>
                        <p className="font-medium">{biller.nickname}</p>
                        <p className="text-sm text-gray-600">{biller.provider_name} - {biller.category}</p>
                        <p className="text-xs text-gray-500">{biller.consumer_id}</p>
                      </div>
                    </div>
                    <div className="flex space-x-2">
                      <Button size="sm" onClick={() => handleBillSelect(biller)}>Pay Bill</Button>
                      <Button size="sm" variant="outline" onClick={() => handleDeleteBiller(biller.id)}>
                        <Trash2 className="h-4 w-4" />
                      </Button>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>
        
        <Card>
          <CardHeader><CardTitle className="text-lg font-semibold">Add New Biller</CardTitle></CardHeader>
          <CardContent>
            <div className="space-y-4">
              {availableBillTypes.map((category) => (
                <div key={category.category}>
                  <h4 className="font-medium mb-2">{category.category}</h4>
                  <div className="space-y-2">
                    {category.bills.map((bill) => {
                      const Icon = bill.icon;
                      return (
                        <div key={bill.id} className="border rounded-lg p-3">
                          <div className="flex items-center space-x-3 mb-2">
                            <div className="bg-gray-100 p-2 rounded-lg"><Icon className="h-4 w-4 text-gray-600" /></div>
                            <span className="font-medium text-sm">{bill.name}</span>
                          </div>
                          <div className="grid grid-cols-2 gap-1">
                            {bill.providers.map((provider) => (
                              <Button key={provider} variant="outline" size="sm" className="text-xs" onClick={() => handleAddBiller(bill, provider)}>
                                {provider}
                              </Button>
                            ))}
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
);
const RechargeView = ({ resetToDashboard, paymentResult, accountBalance, rechargeForm, setRechargeForm, processRecharge, isProcessing, handlePlanRecharge }) => {
    const canRechargePlan = rechargeForm.provider && rechargeForm.mobileNumber.length === 10;
    return (
        <div className="space-y-6 max-w-4xl mx-auto">
            <div className="flex items-center space-x-4">
                <Button variant="ghost" onClick={resetToDashboard}><ArrowLeft className="h-4 w-4 mr-2" />Back to Dashboard</Button>
                <h2 className="text-2xl font-bold">Mobile & DTH Recharge</h2>
            </div>
            {paymentResult && (
                <Alert className={paymentResult.success ? "border-green-200 bg-green-50" : "border-red-200 bg-red-50"}>
                    <div className="flex items-center">
                        {paymentResult.success ? <CheckCircle className="h-4 w-4 text-green-600 mr-2" /> : <AlertCircle className="h-4 w-4 text-red-600 mr-2" />}
                        <AlertDescription className={paymentResult.success ? "text-green-800" : "text-red-800"}>
                            {paymentResult.message}
                            {paymentResult.transactionId && <div className="mt-2"><strong>Transaction ID:</strong> {paymentResult.transactionId}</div>}
                        </AlertDescription>
                    </div>
                    {paymentResult.success && (
                        <div className="mt-4 flex space-x-2">
                            <Button size="sm" onClick={resetToDashboard}>Back to Dashboard</Button>
                            <Button size="sm" variant="outline" onClick={() => generateReceipt(paymentResult.payment)}><Download className="h-4 w-4 mr-2" />Download Receipt</Button>
                        </div>
                    )}
                </Alert>
            )}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <Card>
                    <CardHeader>
                        <CardTitle>Recharge Details</CardTitle>
                        <div className="flex items-center justify-between"><span>Available Balance: ₹{accountBalance.toLocaleString('en-IN')}</span></div>
                    </CardHeader>
                    <CardContent className="space-y-4">
                        <div>
                            <Label>Plan Type</Label>
                            <div className="flex space-x-4 mt-2">
                                <Button variant={rechargeForm.planType === 'prepaid' ? 'default' : 'outline'} size="sm" onClick={() => setRechargeForm(prev => ({ ...prev, planType: 'prepaid' }))}>Prepaid</Button>
                                <Button variant={rechargeForm.planType === 'postpaid' ? 'default' : 'outline'} size="sm" onClick={() => setRechargeForm(prev => ({ ...prev, planType: 'postpaid' }))}>Postpaid</Button>
                            </div>
                        </div>
                        <div>
                            <Label htmlFor="provider">Service Provider</Label>
                            <select id="provider" className="w-full p-2 border rounded-md mt-2" value={rechargeForm.provider} onChange={(e) => setRechargeForm(prev => ({ ...prev, provider: e.target.value, amount: '' }))}>
                                <option value="">Select Provider</option><option value="Airtel">Airtel</option><option value="Jio">Jio</option><option value="Vi">Vi</option><option value="BSNL">BSNL</option>
                            </select>
                        </div>
                        <div>
                            <Label htmlFor="mobileNumber">Mobile Number</Label>
                            <Input id="mobileNumber" value={rechargeForm.mobileNumber} onChange={(e) => setRechargeForm(prev => ({ ...prev, mobileNumber: e.target.value.replace(/\D/g, '') }))} placeholder="Enter 10-digit mobile number" maxLength={10} />
                        </div>
                        <div>
                            <Label htmlFor="amount">Or Enter Amount (₹)</Label>
                            <Input id="amount" type="number" value={rechargeForm.amount} onChange={(e) => setRechargeForm(prev => ({ ...prev, amount: e.target.value }))} placeholder="Enter custom amount" />
                        </div>
                        <Button className="w-full" onClick={() => handlePlanRecharge({ amount: rechargeForm.amount })} disabled={isProcessing || !canRechargePlan || !rechargeForm.amount || parseFloat(rechargeForm.amount) <= 0}>
                            {isProcessing ? "Processing..." : `Recharge ₹${parseFloat(rechargeForm.amount || 0).toLocaleString('en-IN')}`}
                        </Button>
                    </CardContent>
                </Card>
                <Card>
                    <CardHeader><CardTitle>Popular Plans</CardTitle></CardHeader>
                    <CardContent>
                        <div className="space-y-3 max-h-96 overflow-y-auto">
                            {rechargePlans[rechargeForm.planType].map((plan) => (
                                <div key={plan.amount} className="border rounded-lg p-3 hover:bg-gray-50">
                                    <div className="flex justify-between items-start gap-4">
                                        <div>
                                            <p className="font-semibold text-lg">₹{plan.amount}</p>
                                            {plan.validity && <p className="text-sm text-gray-600">Validity: {plan.validity}</p>}
                                            {plan.data && <p className="text-sm text-gray-600">Data: {plan.data}</p>}
                                            {plan.features && <p className="text-sm text-gray-600">Features: {plan.features}</p>}
                                            <p className="text-xs text-gray-500 mt-1">{plan.description}</p>
                                        </div>
                                        <Button size="sm" variant="default" onClick={() => handlePlanRecharge(plan)} disabled={isProcessing || !canRechargePlan}>
                                            Recharge
                                        </Button>
                                    </div>
                                </div>
                            ))}
                        </div>
                    </CardContent>
                </Card>
            </div>
        </div>
    )
};
const PaymentView = ({ setCurrentView, paymentResult, selectedBill, paymentForm, setPaymentForm, accountBalance, processPayment, isProcessing, resetToDashboard }) => (
    <div className="max-w-md mx-auto space-y-6">
      <div className="flex items-center space-x-4"><Button variant="ghost" onClick={() => setCurrentView('billTypes')}><ArrowLeft className="h-4 w-4 mr-2" />Back to Bill Types</Button></div>
      {paymentResult ? (
        <Card>
          <CardHeader><CardTitle className="flex items-center">{paymentResult.success ? <CheckCircle className="h-6 w-6 mr-2 text-green-600" /> : <AlertCircle className="h-6 w-6 mr-2 text-red-600" />}Payment Status</CardTitle></CardHeader>
          <CardContent>
            <Alert className={paymentResult.success ? "border-green-200 bg-green-50" : "border-red-200 bg-red-50"}>
              <AlertDescription className={paymentResult.success ? "text-green-800" : "text-red-800"}>{paymentResult.message}</AlertDescription>
            </Alert>
            {paymentResult.success && paymentResult.payment && (
              <div className="mt-4 space-y-2 text-sm">
                <p><strong>Transaction ID:</strong> {paymentResult.transactionId}</p>
                <p><strong>Service:</strong> {paymentResult.payment.service}</p>
                <p><strong>Amount:</strong> ₹{paymentResult.payment.amount.toLocaleString('en-IN')}</p>
                <p><strong>Date:</strong> {new Date(paymentResult.payment.date).toLocaleDateString('en-IN')}</p>
              </div>
            )}
            <div className="mt-6 flex space-x-2">
              <Button className="w-full" onClick={resetToDashboard}>Back to Dashboard</Button>
              {paymentResult.success && (<Button className="w-full" variant="outline" onClick={() => generateReceipt(paymentResult.payment)}><Download className="h-4 w-4 mr-2" />Download Receipt</Button>)}
            </div>
          </CardContent>
        </Card>
      ) : (
        <Card>
          <CardHeader>
            <CardTitle>{selectedBill?.nickname || selectedBill?.category} - {selectedBill?.provider_name}</CardTitle>
            <div className="flex items-center justify-between"><span>Available Balance: ₹{accountBalance.toLocaleString('en-IN')}</span></div>
          </CardHeader>
          <CardContent className="space-y-4">
            <div><Label htmlFor="billNumber">Consumer ID</Label><Input id="billNumber" value={selectedBill?.consumer_id} disabled /></div>
            <div><Label htmlFor="amount">Amount (₹)</Label><Input id="amount" type="number" value={paymentForm.amount} onChange={(e) => setPaymentForm(prev => ({ ...prev, amount: e.target.value }))} placeholder="Enter amount to pay" /></div>
            <Button className="w-full" onClick={processPayment} disabled={isProcessing || !paymentForm.amount || parseFloat(paymentForm.amount) <= 0}>
              {isProcessing ? "Processing Payment..." : `Pay ₹${parseFloat(paymentForm.amount || 0).toLocaleString('en-IN')}`}
            </Button>
          </CardContent>
        </Card>
      )}
    </div>
);
const HistoryView = ({ resetToDashboard, paymentHistory, clearAllData }) => {
    const [searchTerm, setSearchTerm] = useState('');
    const [filterType, setFilterType] = useState('all');
    const filteredHistory = useMemo(() => paymentHistory
      .filter(p => {
        // First filter: Only show bill payment related transactions
        const billPaymentTypes = ['BILL_PAYMENT', 'RECHARGE', 'TAX_DIRECT', 'TAX_GST', 'TAX_STATE'];
        if (!billPaymentTypes.includes(p.type)) return false;
        
        // Second filter: Apply user selected filter
        if (filterType === 'all') return true;
        if (filterType === 'bill') return p.type === 'BILL_PAYMENT';
        if (filterType === 'recharge') return p.type === 'RECHARGE';
        if (filterType === 'tax') return ['TAX_DIRECT', 'TAX_GST', 'TAX_STATE'].includes(p.type);
        return p.type === filterType;
      })
      .filter(p => {
        const description = p.description || '';
        const counterparty = p.counterparty || '';
        const id = p.id || '';
        return description.toLowerCase().includes(searchTerm.toLowerCase()) || 
               counterparty.toLowerCase().includes(searchTerm.toLowerCase()) || 
               id.toLowerCase().includes(searchTerm.toLowerCase());
      }), 
      [paymentHistory, searchTerm, filterType]
    );

    return (
      <div className="space-y-6">
        <div className="flex items-center space-x-4"><Button variant="ghost" onClick={resetToDashboard}><ArrowLeft className="h-4 w-4 mr-2" />Back to Dashboard</Button><h2 className="text-2xl font-bold">Payment History</h2></div>
        <Card>
          <CardHeader>
            <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
              <Input placeholder="Search by service, bill number, or TXN ID..." value={searchTerm} onChange={(e) => setSearchTerm(e.target.value)} className="max-w-sm" />
              <div className='flex items-center gap-4'>
                <div className="flex items-center space-x-2">
                  <Button variant={filterType === 'all' ? 'default' : 'outline'} onClick={() => setFilterType('all')}>All</Button>
                  <Button variant={filterType === 'bill' ? 'default' : 'outline'} onClick={() => setFilterType('bill')}>Bills</Button>
                  <Button variant={filterType === 'recharge' ? 'default' : 'outline'} onClick={() => setFilterType('recharge')}>Recharges</Button>
                </div>
              </div>
            </div>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {filteredHistory.length > 0 ? (
                filteredHistory.map(payment => (
                  <div key={payment.id} className="border rounded-lg p-4 flex flex-col md:flex-row md:items-center justify-between gap-4">
                    <div className="flex items-center space-x-4">
                      <div className={`p-2 rounded-lg ${payment.type === 'BILL_PAYMENT' ? 'bg-blue-100' : 'bg-orange-100'}`}>{payment.type === 'BILL_PAYMENT' ? <Receipt className="h-5 w-5 text-blue-600" /> : <Smartphone className="h-5 w-5 text-orange-600" />}</div>
                      <div>
                        <p className="font-semibold">{payment.description}</p>
                        <p className="text-sm text-gray-500">Provider: {payment.counterparty}</p>
                        <p className="text-xs text-gray-400">TXN ID: {payment.id}</p>
                      </div>
                    </div>
                    <div className="text-right">
                      <p className="font-bold text-lg">₹{payment.amount.toLocaleString('en-IN')}</p>
                      <p className="text-sm text-gray-500">{new Date(payment.createdAt).toLocaleDateString('en-IN')}</p>
                      <Badge className={payment.status === 'COMPLETED' ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'}>{payment.status}</Badge>
                    </div>
                    <Button variant="outline" size="sm" onClick={() => generateReceipt(payment)}><Download className="h-4 w-4 mr-2" />Receipt</Button>
                  </div>
                ))
              ) : (<div className="text-center py-12 text-gray-500"><p>No transactions found.</p></div>)}
            </div>
          </CardContent>
        </Card>
      </div>
    );
};
const AutoPayView = ({ resetToDashboard, autoPayRules, registeredBillers, addAutoPayRule, toggleAutoPayRule, deleteAutoPayRule, isLoadingRules, isLoadingBillers, selectedBillerForAutoPay, setSelectedBillerForAutoPay, autoPayMaxAmount, setAutoPayMaxAmount }) => (
    <div className="space-y-6">
      <div className="flex items-center space-x-4"><Button variant="ghost" onClick={resetToDashboard}><ArrowLeft className="h-4 w-4 mr-2" />Back to Dashboard</Button><h2 className="text-2xl font-bold">Auto Pay Setup</h2></div>
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card>
          <CardHeader><CardTitle>Add New Auto Pay Rule</CardTitle></CardHeader>
          <CardContent className="space-y-4">
            {isLoadingBillers ? (
              <div className="space-y-3">
                <div className="animate-pulse h-10 bg-gray-200 rounded"></div>
                <div className="animate-pulse h-10 bg-gray-200 rounded"></div>
              </div>
            ) : registeredBillers.length === 0 ? (
              <div className="text-center py-8 text-gray-500">
                <Shield className="h-12 w-12 mx-auto mb-4 text-gray-300" />
                <p>No billers registered</p>
                <p className="text-sm">Add billers first to set up auto-pay</p>
              </div>
            ) : (
              <>
                <div>
                  <Label htmlFor="autoPayBiller">Select Biller</Label>
                  <select id="autoPayBiller" className="w-full p-2 border rounded-md mt-2" onChange={(e) => {
                    const billerId = e.target.value;
                    const biller = registeredBillers.find(b => b.id === billerId);
                    setSelectedBillerForAutoPay(biller);
                  }}>
                    <option value="">Select a biller</option>
                    {registeredBillers.map(biller => (
                      <option key={biller.id} value={biller.id}>
                        {biller.nickname} - {biller.provider_name}
                      </option>
                    ))}
                  </select>
                </div>
                <div><Label htmlFor="autoPayMaxAmount">Max Amount (₹)</Label><Input id="autoPayMaxAmount" type="number" placeholder="e.g., 3000" value={autoPayMaxAmount} onChange={e => setAutoPayMaxAmount(e.target.value)} /></div>
                <Button onClick={() => addAutoPayRule(selectedBillerForAutoPay?.id, autoPayMaxAmount)} className="w-full" disabled={!selectedBillerForAutoPay || !autoPayMaxAmount}><Plus className="h-4 w-4 mr-2" />Add Rule</Button>
              </>
            )}
          </CardContent>
        </Card>
        <Card>
          <CardHeader><CardTitle>Your Auto Pay Rules</CardTitle></CardHeader>
          <CardContent>
            {isLoadingRules ? (
              <div className="space-y-3">
                {[1,2].map(i => (
                  <div key={i} className="animate-pulse">
                    <div className="h-20 bg-gray-200 rounded-lg"></div>
                  </div>
                ))}
              </div>
            ) : autoPayRules.length === 0 ? (
              <p className="text-gray-500">No auto pay rules set up.</p>
            ) : (
              <div className="space-y-4">
                {autoPayRules.map(rule => (
                  <div key={rule.id} className="border p-4 rounded-lg">
                    <div className="flex justify-between items-start">
                      <div>
                        <p className="font-semibold">{rule.nickname} - {rule.provider_name}</p>
                        <p className="text-sm text-gray-600">Consumer ID: {rule.consumer_id}</p>
                        <p className="text-sm text-gray-600">Max Amount: ₹{rule.max_amount.toLocaleString('en-IN')}</p>
                        <p className="text-sm text-gray-600">Created: {new Date(rule.created_at).toLocaleDateString('en-IN')}</p>
                      </div>
                      <div className="flex flex-col items-end gap-2">
                        <Badge className={rule.enabled ? 'bg-green-500' : 'bg-gray-500'}>{rule.enabled ? 'Active' : 'Paused'}</Badge>
                        <div className="flex space-x-1">
                          <Button size="icon" variant="outline" onClick={() => toggleAutoPayRule(rule.id, !rule.enabled)}><Settings className="h-4 w-4" /></Button>
                          <Button size="icon" variant="destructive" onClick={() => deleteAutoPayRule(rule.id)}><Trash2 className="h-4 w-4" /></Button>
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
);

// --- MAIN COMPONENT ---
export default function BillPayments() {
  const { user, fetchAndSetProfile } = useAuth();
  const { addNotification } = useNotifications();
  const navigate = useNavigate();
  const [currentView, setCurrentView] = useState('dashboard');
  const [selectedBill, setSelectedBill] = useState<Biller | null>(null);
  
  // --- API STATE ---
  const [registeredBillers, setRegisteredBillers] = useState<Biller[]>([]);
  const [paymentHistory, setPaymentHistory] = useState([]);
  const [autoPayRules, setAutoPayRules] = useState<AutoPayRule[]>([]);
  
  // --- LOADING STATES ---
  const [isLoadingBillers, setIsLoadingBillers] = useState(false);
  const [isLoadingHistory, setIsLoadingHistory] = useState(false);
  const [isLoadingRules, setIsLoadingRules] = useState(false);
  
  // --- AUTOPAY FORM STATE ---
  const [selectedBillerForAutoPay, setSelectedBillerForAutoPay] = useState<Biller | null>(null);
  const [autoPayMaxAmount, setAutoPayMaxAmount] = useState('');
  
  // --- FORM STATE ---
  const [paymentForm, setPaymentForm] = useState({ amount: '' });
  const [rechargeForm, setRechargeForm] = useState({ provider: '', mobileNumber: '', amount: '', planType: 'prepaid' });
  const [isProcessing, setIsProcessing] = useState(false);
  const [paymentResult, setPaymentResult] = useState(null);
  const [addBillerForm, setAddBillerForm] = useState({ provider_name: '', category: '', consumer_id: '', nickname: '' });
  const [showAddBillerModal, setShowAddBillerModal] = useState(false);
  
  // --- LOAD DATA ON MOUNT ---
  useEffect(() => {
    if (user) {
      loadRegisteredBillers();
      loadPaymentHistory();
      loadAutoPayRules();
    }
  }, [user]);
  
  const loadRegisteredBillers = async () => {
    setIsLoadingBillers(true);
    try {
      const response = await billPaymentsApi.getRegisteredBillers();
      if (response.success) {
        setRegisteredBillers(response.billers || []);
      }
    } catch (error) {
      console.error('Error loading billers:', error);
    } finally {
      setIsLoadingBillers(false);
    }
  };
  
  const loadPaymentHistory = async () => {
    setIsLoadingHistory(true);
    try {
      const response = await billPaymentsApi.getPaymentHistory();
      if (response.success) {
        setPaymentHistory(response.transactions || []);
      }
    } catch (error) {
      console.error('Error loading payment history:', error);
    } finally {
      setIsLoadingHistory(false);
    }
  };
  
  const loadAutoPayRules = async () => {
    setIsLoadingRules(true);
    try {
      const response = await billPaymentsApi.getAutoPayRules();
      if (response.success) {
        setAutoPayRules(response.rules || []);
      }
    } catch (error) {
      console.error('Error loading auto pay rules:', error);
    } finally {
      setIsLoadingRules(false);
    }
  };

  // --- HANDLERS ---
  const handleCategoryClick = (category) => {
    if (category.title === "Payment History") setCurrentView('history');
    else if (category.title === "Pay Bills") setCurrentView('billTypes');
    else if (category.title === "Topup Recharge") setCurrentView('recharge');
    else if (category.title === "Auto Pay Setup") setCurrentView('autopay');
  };

  const handleBillSelect = (biller: Biller) => {
    setSelectedBill(biller);
    setPaymentForm({ amount: '' });
    setCurrentView('payment');
  };
  
  const handleAddBiller = (bill: any, provider: string) => {
    setAddBillerForm({
      provider_name: provider,
      category: bill.name,
      consumer_id: '',
      nickname: `My ${provider} ${bill.name}`
    });
    setShowAddBillerModal(true);
  };
  
  const confirmAddBiller = async () => {
    if (!addBillerForm.provider_name || !addBillerForm.category || !addBillerForm.consumer_id || !addBillerForm.nickname) {
      alert('Please fill all fields');
      return;
    }
    
    try {
      const response = await billPaymentsApi.addBiller(addBillerForm);
      if (response.success) {
        await loadRegisteredBillers();
        setShowAddBillerModal(false);
        setAddBillerForm({ provider_name: '', category: '', consumer_id: '', nickname: '' });
      } else {
        alert(response.error || 'Failed to add biller');
      }
    } catch (error) {
      console.error('Error adding biller:', error);
      alert('Error adding biller');
    }
  };
  
  const handleDeleteBiller = async (billerId: string) => {
    if (!confirm('Are you sure you want to delete this biller?')) return;
    
    try {
      const response = await billPaymentsApi.deleteBiller(billerId);
      if (response.success) {
        await loadRegisteredBillers();
      } else {
        alert(response.error || 'Failed to delete biller');
      }
    } catch (error) {
      console.error('Error deleting biller:', error);
      alert('Error deleting biller');
    }
  };
  
  const processPayment = useCallback(async () => {
    if (!selectedBill) return;
    
    setIsProcessing(true);
    const paymentAmount = parseFloat(paymentForm.amount);
    
    if (paymentAmount > (user?.balance || 0)) {
      setPaymentResult({ success: false, message: "Insufficient balance." });
      setIsProcessing(false);
      return;
    }
    
    try {
      const response = await billPaymentsApi.payBill(selectedBill.id, paymentAmount, selectedBill.provider_name, addNotification);
      if (response.success) {
        setPaymentResult({ 
          success: true, 
          message: response.message || "Payment processed successfully!",
          transactionId: `TXN${Date.now()}`,
          payment: {
            id: Date.now(),
            service: `${selectedBill.nickname} - ${selectedBill.provider_name}`,
            amount: paymentAmount,
            date: new Date().toISOString(),
            status: "Success",
            billNumber: selectedBill.consumer_id,
            provider: selectedBill.provider_name,
            transactionId: `TXN${Date.now()}`,
            customerName: user?.firstName + ' ' + user?.lastName,
            type: 'bill'
          }
        });
        // Refresh user balance and payment history
        await fetchAndSetProfile();
        await loadPaymentHistory();
        
        // Check for anomalies after successful bill payment
        const anomalyDetected = await checkAnomalyAfterTransaction(
          'bill_payment',
          paymentAmount,
          navigate,
          '/bills'
        );
        
        if (anomalyDetected) {
          // User will be redirected to anomaly verification page
          return;
        }
      } else {
        setPaymentResult({ success: false, message: response.error || "Payment failed" });
      }
    } catch (error) {
      console.error('Payment error:', error);
      setPaymentResult({ success: false, message: "Payment failed. Please try again." });
    } finally {
      setIsProcessing(false);
    }
  }, [selectedBill, paymentForm.amount, user, fetchAndSetProfile]);

  const processRecharge = useCallback(async (amount) => {
    if (!rechargeForm.provider || !rechargeForm.mobileNumber) {
      setPaymentResult({ success: false, message: "Please fill in all required fields." });
      return;
    }

    setIsProcessing(true);
    setPaymentResult(null); // Clear previous results
    const rechargeAmount = parseFloat(amount);
    
    if (rechargeAmount > (user?.balance || 0)) {
      setPaymentResult({ success: false, message: "Insufficient balance." });
      setIsProcessing(false);
      return;
    }

    try {
      const response = await billPaymentsApi.processRecharge({
        provider: rechargeForm.provider,
        mobile_number: rechargeForm.mobileNumber,
        amount: rechargeAmount,
        plan_type: rechargeForm.planType
      }, addNotification);

      if (response.success) {
        setPaymentResult({ 
          success: true, 
          message: response.message || "Recharge completed successfully!",
          transactionId: response.transaction_id,
          payment: {
            id: response.transaction_id,
            service: `${rechargeForm.planType === 'prepaid' ? 'Prepaid' : 'Postpaid'} Recharge - ${rechargeForm.provider}`,
            amount: rechargeAmount,
            date: new Date().toISOString(),
            status: "Success",
            billNumber: rechargeForm.mobileNumber,
            provider: rechargeForm.provider,
            transactionId: response.transaction_id,
            customerName: user?.firstName + ' ' + user?.lastName,
            type: 'recharge'
          }
        });
        // Refresh user balance and payment history
        await fetchAndSetProfile();
        await loadPaymentHistory();
        setRechargeForm(prev => ({...prev, amount: ''})); // Clear amount field
      } else {
        setPaymentResult({ success: false, message: response.error || "Recharge failed" });
      }
    } catch (error) {
      console.error('Recharge error:', error);
      setPaymentResult({ success: false, message: "Recharge failed. Please try again." });
    } finally {
      setIsProcessing(false);
    }
  }, [user?.balance, rechargeForm, fetchAndSetProfile]);

  const handlePlanRecharge = useCallback((plan) => {
      setRechargeForm(prev => ({...prev, amount: plan.amount.toString()}));
      processRecharge(plan.amount);
  }, [processRecharge]);

  const addAutoPayRule = async (billerId: string, maxAmount: string) => {
    if (!billerId || !maxAmount) return;
    
    try {
      const response = await billPaymentsApi.addAutoPayRule(billerId, parseFloat(maxAmount));
      if (response.success) {
        await loadAutoPayRules();
        setSelectedBillerForAutoPay(null);
        setAutoPayMaxAmount('');
      } else {
        alert(response.error || 'Failed to add auto pay rule');
      }
    } catch (error) {
      console.error('Error adding auto pay rule:', error);
      alert('Error adding auto pay rule');
    }
  };

  const toggleAutoPayRule = async (ruleId: string, enabled: boolean) => {
    try {
      const response = await billPaymentsApi.toggleAutoPayRule(ruleId, enabled);
      if (response.success) {
        await loadAutoPayRules();
      } else {
        alert(response.error || 'Failed to update auto pay rule');
      }
    } catch (error) {
      console.error('Error toggling auto pay rule:', error);
      alert('Error updating auto pay rule');
    }
  };
  
  const deleteAutoPayRule = async (ruleId: string) => {
    if (!confirm('Are you sure you want to delete this auto pay rule?')) return;
    
    try {
      const response = await billPaymentsApi.deleteAutoPayRule(ruleId);
      if (response.success) {
        await loadAutoPayRules();
      } else {
        alert(response.error || 'Failed to delete auto pay rule');
      }
    } catch (error) {
      console.error('Error deleting auto pay rule:', error);
      alert('Error deleting auto pay rule');
    }
  };

  const resetToDashboard = () => {
    setCurrentView('dashboard');
    setSelectedBill(null);
    setPaymentForm({ amount: '' });
    setRechargeForm({ provider: '', mobileNumber: '', amount: '', planType: 'prepaid' });
    setPaymentResult(null);
  };

  const clearAllData = () => {
      if (window.confirm("Are you sure you want to clear all local data? This will not affect your server data.")) {
          setPaymentHistory([]);
          setAutoPayRules([]);
          setRegisteredBillers([]);
      }
  };

  const renderView = () => {
    switch (currentView) {
      case 'dashboard': return <DashboardView accountBalance={user?.balance || 0} paymentHistory={paymentHistory} handleCategoryClick={handleCategoryClick} setCurrentView={setCurrentView} isLoadingHistory={isLoadingHistory} />;
      case 'billTypes': return <BillTypesView resetToDashboard={resetToDashboard} handleBillSelect={handleBillSelect} registeredBillers={registeredBillers} handleAddBiller={handleAddBiller} handleDeleteBiller={handleDeleteBiller} isLoadingBillers={isLoadingBillers} />;
      case 'recharge': return <RechargeView resetToDashboard={resetToDashboard} paymentResult={paymentResult} accountBalance={user?.balance || 0} rechargeForm={rechargeForm} setRechargeForm={setRechargeForm} processRecharge={processRecharge} isProcessing={isProcessing} handlePlanRecharge={handlePlanRecharge} />;
      case 'payment': return <PaymentView setCurrentView={setCurrentView} paymentResult={paymentResult} selectedBill={selectedBill} paymentForm={paymentForm} setPaymentForm={setPaymentForm} accountBalance={user?.balance || 0} processPayment={processPayment} isProcessing={isProcessing} resetToDashboard={resetToDashboard} />;
      case 'history': return <HistoryView resetToDashboard={resetToDashboard} paymentHistory={paymentHistory} clearAllData={clearAllData} />;
      case 'autopay': return <AutoPayView resetToDashboard={resetToDashboard} autoPayRules={autoPayRules} registeredBillers={registeredBillers} addAutoPayRule={addAutoPayRule} toggleAutoPayRule={toggleAutoPayRule} deleteAutoPayRule={deleteAutoPayRule} isLoadingRules={isLoadingRules} isLoadingBillers={isLoadingBillers} selectedBillerForAutoPay={selectedBillerForAutoPay} setSelectedBillerForAutoPay={setSelectedBillerForAutoPay} autoPayMaxAmount={autoPayMaxAmount} setAutoPayMaxAmount={setAutoPayMaxAmount} />;
      default: return <DashboardView accountBalance={user?.balance || 0} paymentHistory={paymentHistory} handleCategoryClick={handleCategoryClick} setCurrentView={setCurrentView} isLoadingHistory={isLoadingHistory} />;
    }
  };

  return (
    <BankingLayout>
      <div className="space-y-6 animate-slide-in">
        {renderView()}
        
        {/* Add Biller Modal */}
        {showAddBillerModal && (
          <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
            <div className="bg-white p-6 rounded-lg max-w-md w-full mx-4">
              <h3 className="text-lg font-semibold mb-4">Add New Biller</h3>
              <div className="space-y-4">
                <div>
                  <Label>Provider</Label>
                  <Input value={addBillerForm.provider_name} disabled className="bg-gray-100" />
                </div>
                <div>
                  <Label>Category</Label>
                  <Input value={addBillerForm.category} disabled className="bg-gray-100" />
                </div>
                <div>
                  <Label>Consumer ID/Bill Number</Label>
                  <Input 
                    value={addBillerForm.consumer_id} 
                    onChange={(e) => setAddBillerForm(prev => ({...prev, consumer_id: e.target.value}))}
                    placeholder="Enter your consumer ID or account number"
                  />
                </div>
                <div>
                  <Label>Nickname</Label>
                  <Input 
                    value={addBillerForm.nickname} 
                    onChange={(e) => setAddBillerForm(prev => ({...prev, nickname: e.target.value}))}
                    placeholder="e.g., Home Electricity"
                  />
                </div>
              </div>
              <div className="flex space-x-2 mt-6">
                <Button onClick={confirmAddBiller} className="flex-1">Add Biller</Button>
                <Button onClick={() => setShowAddBillerModal(false)} variant="outline" className="flex-1">Cancel</Button>
              </div>
            </div>
          </div>
        )}
      </div>
    </BankingLayout>
  );
}