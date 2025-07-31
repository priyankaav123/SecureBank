import React, { useState, useEffect } from 'react';
import { BankingLayout } from '@/components/layout/BankingLayout';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { toast } from "sonner";
import { useAuth } from '@/contexts/AuthContext';
import { useNotifications } from '@/contexts/NotificationContext';
import { taxApi, TaxPayment } from '@/services/taxapi';
import {
  FileText,
  Building,
  Receipt,
  Calculator,
  Download,
  Shield,
  Clock,
  ChevronRight,
  ArrowLeft,
  Plus,
  CheckCircle,
  FileDigit,
  Landmark,
  Percent,
  UserCheck
} from 'lucide-react';

// --- DATA & MOCKS ---
const taxServices = [
  { id: 'direct', title: "Direct Taxes", description: "Income tax, TDS, Advance Tax", icon: UserCheck, color: "text-purple-600", bgColor: "bg-purple-100" },
  { id: 'gst', title: "Goods & Services Tax (GST)", description: "Pay your GST liabilities", icon: Percent, color: "text-green-600", bgColor: "bg-green-100" },
  { id: 'state', title: "State Govt. Taxes", description: "Property, Professional tax, etc.", icon: Landmark, color: "text-blue-600", bgColor: "bg-blue-100" },
  { id: 'history', title: "Challans & History", description: "Download receipts & view history", icon: FileDigit, color: "text-orange-600", bgColor: "bg-orange-100" },
];

// --- VIEW COMPONENTS FOR EACH TAX TYPE ---

const DirectTaxView = ({ onBack, onSubmit, isLoading }) => {
  const [form, setForm] = useState({ pan: '', assessmentYear: '2024-25', taxType: '0021', paymentType: '300', amount: '' });
  const handleSubmit = () => {
    if (!form.pan || form.pan.length !== 10 || !form.amount || +form.amount <= 0) {
      toast.error("Invalid Details", { description: "Please enter a valid 10-digit PAN and a positive amount." });
      return;
    }
    onSubmit(form);
  };
  return (
    <div className="space-y-6">
      <Button variant="ghost" onClick={onBack}><ArrowLeft className="mr-2 h-4 w-4" /> Back to e-Tax Dashboard</Button>
      <Card>
        <CardHeader>
          <CardTitle>Pay Direct Tax (Income Tax, TDS)</CardTitle>
          <p className="text-sm text-muted-foreground">Challan No./ITNS 280</p>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid md:grid-cols-2 gap-4">
            <div><Label htmlFor="pan">PAN / TAN *</Label><Input id="pan" value={form.pan} onChange={e => setForm({...form, pan: e.target.value.toUpperCase()})} maxLength={10} placeholder="Enter 10-digit PAN"/></div>
            <div><Label htmlFor="ay">Assessment Year *</Label><Select value={form.assessmentYear} onValueChange={v => setForm({...form, assessmentYear:v})}><SelectTrigger><SelectValue/></SelectTrigger><SelectContent><SelectItem value="2024-25">2024-25</SelectItem><SelectItem value="2023-24">2023-24</SelectItem></SelectContent></Select></div>
            <div><Label htmlFor="tax-type">Tax Applicable *</Label><Select value={form.taxType} onValueChange={v => setForm({...form, taxType:v})}><SelectTrigger><SelectValue/></SelectTrigger><SelectContent><SelectItem value="0021">(0021) Income-Tax Other Than Companies</SelectItem><SelectItem value="0020">(0020) Corporation Tax</SelectItem></SelectContent></Select></div>
            <div><Label htmlFor="payment-type">Type of Payment *</Label><Select value={form.paymentType} onValueChange={v => setForm({...form, paymentType:v})}><SelectTrigger><SelectValue/></SelectTrigger><SelectContent><SelectItem value="100">(100) Advance Tax</SelectItem><SelectItem value="300">(300) Self-Assessment Tax</SelectItem><SelectItem value="400">(400) Tax on Regular Assessment</SelectItem></SelectContent></Select></div>
          </div>
          <div><Label htmlFor="amount">Amount (₹) *</Label><Input id="amount" type="number" placeholder="Enter total tax amount" value={form.amount} onChange={e => setForm({...form, amount: e.target.value})}/></div>
          <Button onClick={handleSubmit} className="w-full" disabled={isLoading}>
            {isLoading ? 'Processing...' : 'Proceed to Pay'}
          </Button>
        </CardContent>
      </Card>
    </div>
  );
};

const GstView = ({ onBack, onSubmit, isLoading }) => {
    const [form, setForm] = useState({ gstin: '', cpin: '', cgst: '', sgst: '', igst: '', cess: '' });
    const total = (+(form.cgst) || 0) + (+(form.sgst) || 0) + (+(form.igst) || 0) + (+(form.cess) || 0);
    const handleSubmit = () => {
        if (!form.gstin || form.gstin.length !== 15 || !form.cpin || total <= 0) {
            toast.error("Invalid Details", { description: "Please enter a valid 15-digit GSTIN, CPIN, and tax amounts." });
            return;
        }
        onSubmit({ ...form, amount: total });
    };
    return (
        <div className="space-y-6">
            <Button variant="ghost" onClick={onBack}><ArrowLeft className="mr-2 h-4 w-4" /> Back to e-Tax Dashboard</Button>
            <Card>
                <CardHeader>
                    <CardTitle>Pay Goods and Services Tax (GST)</CardTitle>
                    <p className="text-sm text-muted-foreground">Enter CPIN generated from the GST Portal to proceed.</p>
                </CardHeader>
                <CardContent className="space-y-4">
                    <div className="grid md:grid-cols-2 gap-4">
                        <div><Label htmlFor="gstin">GSTIN *</Label><Input id="gstin" value={form.gstin} onChange={e => setForm({...form, gstin: e.target.value.toUpperCase()})} maxLength={15} placeholder="15-digit GST Identification Number"/></div>
                        <div><Label htmlFor="cpin">CPIN *</Label><Input id="cpin" type="number" value={form.cpin} onChange={e => setForm({...form, cpin: e.target.value})} placeholder="14-digit Common Portal ID"/></div>
                    </div>
                    <Label>Tax Amount Breakup (₹) *</Label>
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                        <div><Label htmlFor="cgst" className="text-xs">CGST</Label><Input id="cgst" type="number" value={form.cgst} onChange={e => setForm({...form, cgst: e.target.value})} placeholder="0.00"/></div>
                        <div><Label htmlFor="sgst" className="text-xs">SGST</Label><Input id="sgst" type="number" value={form.sgst} onChange={e => setForm({...form, sgst: e.target.value})} placeholder="0.00"/></div>
                        <div><Label htmlFor="igst" className="text-xs">IGST</Label><Input id="igst" type="number" value={form.igst} onChange={e => setForm({...form, igst: e.target.value})} placeholder="0.00"/></div>
                        <div><Label htmlFor="cess" className="text-xs">Cess</Label><Input id="cess" type="number" value={form.cess} onChange={e => setForm({...form, cess: e.target.value})} placeholder="0.00"/></div>
                    </div>
                    <div className="border-t pt-4 flex justify-between items-center">
                        <span className="font-semibold">Total Payable Amount:</span>
                        <span className="text-2xl font-bold">₹{total.toLocaleString('en-IN')}</span>
                    </div>
                    <Button onClick={handleSubmit} className="w-full" disabled={isLoading}>
                      {isLoading ? 'Processing...' : 'Confirm and Pay GST'}
                    </Button>
                </CardContent>
            </Card>
        </div>
    );
};

const StateTaxView = ({ onBack, onSubmit, isLoading }) => {
    const [form, setForm] = useState({ state: '', municipality: '', service: '', consumerId: '', amount: '' });
    const handleSubmit = () => {
        if (!form.state || !form.municipality || !form.service || !form.consumerId || !form.amount || +form.amount <= 0) {
            toast.error("All Fields Required", { description: "Please complete all steps to pay your state tax." });
            return;
        }
        onSubmit(form);
    }
    return (
        <div className="space-y-6">
            <Button variant="ghost" onClick={onBack}><ArrowLeft className="mr-2 h-4 w-4" /> Back to e-Tax Dashboard</Button>
            <Card>
                <CardHeader><CardTitle>Pay State Government Taxes</CardTitle><p className="text-sm text-muted-foreground">Select your state, municipality, and service to continue.</p></CardHeader>
                <CardContent className="space-y-4">
                    <div className="grid md:grid-cols-3 gap-4">
                        <div><Label>1. Select State *</Label><Select value={form.state} onValueChange={v => setForm({...form, state:v})}><SelectTrigger><SelectValue placeholder="Select State..."/></SelectTrigger><SelectContent><SelectItem value="Maharashtra">Maharashtra</SelectItem><SelectItem value="Karnataka">Karnataka</SelectItem><SelectItem value="Delhi">Delhi</SelectItem></SelectContent></Select></div>
                        <div><Label>2. Select Municipality *</Label><Select value={form.municipality} onValueChange={v => setForm({...form, municipality:v})} disabled={!form.state}><SelectTrigger><SelectValue placeholder="Select Municipality..."/></SelectTrigger><SelectContent><SelectItem value="Mumbai Corp">Mumbai Corporation</SelectItem><SelectItem value="BBMP">BBMP (Bengaluru)</SelectItem><SelectItem value="MCD">MCD (Delhi)</SelectItem></SelectContent></Select></div>
                        <div><Label>3. Select Service *</Label><Select value={form.service} onValueChange={v => setForm({...form, service:v})} disabled={!form.municipality}><SelectTrigger><SelectValue placeholder="Select Service..."/></SelectTrigger><SelectContent><SelectItem value="Property Tax">Property Tax</SelectItem><SelectItem value="Professional Tax">Professional Tax</SelectItem><SelectItem value="Water Tax">Water Tax</SelectItem></SelectContent></Select></div>
                    </div>
                    {form.service && (
                        <div className="grid md:grid-cols-2 gap-4 animate-slide-in">
                            <div><Label htmlFor="consumerId">Property ID / Consumer No. *</Label><Input id="consumerId" value={form.consumerId} onChange={e => setForm({...form, consumerId: e.target.value})} placeholder="Enter your unique ID"/></div>
                            <div><Label htmlFor="state-amount">Amount to Pay (₹) *</Label><Input id="state-amount" type="number" value={form.amount} onChange={e => setForm({...form, amount: e.target.value})} placeholder="Enter amount"/></div>
                        </div>
                    )}
                    <Button onClick={handleSubmit} className="w-full" disabled={!form.service || isLoading}>
                      {isLoading ? 'Processing...' : 'Fetch Bill & Pay'}
                    </Button>
                </CardContent>
            </Card>
        </div>
    );
}

// --- MAIN E-TAX COMPONENT ---
export default function ETax() {
  const { user, fetchAndSetProfile } = useAuth();
  const { addNotification } = useNotifications();
  const [currentView, setCurrentView] = useState('dashboard');
  const [taxPayments, setTaxPayments] = useState<TaxPayment[]>([]);
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    loadTaxHistory();
  }, []);

  const loadTaxHistory = async () => {
    try {
      const response = await taxApi.getTaxHistory();
      if (response.success && response.tax_payments) {
        setTaxPayments(response.tax_payments);
      }
    } catch (error) {
      console.error('Error loading tax history:', error);
    }
  };

  const handleDownloadChallan = async (taxPaymentId: string) => {
    setIsLoading(true);
    try {
      const response = await taxApi.downloadChallan(taxPaymentId);
      if (response.success && response.challan) {
        // Create and download the challan as JSON file
        const dataStr = JSON.stringify(response.challan, null, 2);
        const dataBlob = new Blob([dataStr], { type: 'application/json' });
        const url = URL.createObjectURL(dataBlob);
        const link = document.createElement('a');
        link.href = url;
        link.download = `challan_${response.challan.challan_number}.json`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);
        toast.success('Challan downloaded successfully!');
      } else {
        toast.error('Download failed', { description: response.error || 'Unknown error' });
      }
    } catch (error) {
      console.error('Download error:', error);
      toast.error('Download failed', { description: 'Network error occurred' });
    } finally {
      setIsLoading(false);
    }
  };

  const handleExportHistory = async () => {
    setIsLoading(true);
    try {
      const response = await taxApi.exportHistory('json');
      if (response.success && response.exportData) {
        // Create and download the history as JSON file
        const dataStr = JSON.stringify(response.exportData, null, 2);
        const dataBlob = new Blob([dataStr], { type: 'application/json' });
        const url = URL.createObjectURL(dataBlob);
        const link = document.createElement('a');
        link.href = url;
        link.download = `tax_history_${new Date().toISOString().split('T')[0]}.json`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);
        toast.success('Tax history exported successfully!');
      } else {
        toast.error('Export failed', { description: response.error || 'Unknown error' });
      }
    } catch (error) {
      console.error('Export error:', error);
      toast.error('Export failed', { description: 'Network error occurred' });
    } finally {
      setIsLoading(false);
    }
  };

  const handleServiceClick = (serviceId) => {
    if (serviceId === 'history') {
        toast.info("Viewing all past payments and challans.");
    }
    setCurrentView(serviceId);
  };
  
  const handlePaymentSubmit = async (formData) => {
    setIsLoading(true);
    
    try {
      let response;
      let paymentType = '';
      
      if (currentView === 'direct') {
        response = await taxApi.payDirectTax({
          ...formData,
          amount: parseFloat(formData.amount)
        }, addNotification);
        paymentType = 'Direct Tax';
      } else if (currentView === 'gst') {
        response = await taxApi.payGst({
          ...formData,
          amount: parseFloat(formData.amount)
        }, addNotification);
        paymentType = 'GST Payment';
      } else if (currentView === 'state') {
        response = await taxApi.payStateTax({
          ...formData,
          amount: parseFloat(formData.amount)
        }, addNotification);
        paymentType = 'State Tax';
      }
      
      if (response?.success) {
        await fetchAndSetProfile(); // Update user balance
        await loadTaxHistory(); // Reload tax history
        toast.success(`${paymentType} Payment Successful!`, { 
          description: `Paid ₹${parseFloat(formData.amount).toLocaleString('en-IN')}` 
        });
        setCurrentView('dashboard');
      } else {
        toast.error('Payment Failed', { description: response?.error || 'Unknown error occurred' });
      }
    } catch (error) {
      console.error('Payment error:', error);
      toast.error('Payment Failed', { description: 'Network error occurred' });
    } finally {
      setIsLoading(false);
    }
  };

  const DashboardView = () => (
    <div className="space-y-6">
      <div className="bg-banking-gradient rounded-lg p-6 text-white animate-slide-in">
        <div className="flex items-center"><FileText className="h-8 w-8 mr-4" /><div><h1 className="text-2xl font-bold">e-Tax Services</h1><p className="text-blue-100">Your one-stop solution for all tax payments.</p></div></div>
      </div>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {taxServices.map((service) => (
          <Card key={service.id} className="hover:shadow-lg transition-shadow cursor-pointer group" onClick={() => handleServiceClick(service.id)}>
            <CardContent className="p-6 flex items-start space-x-4">
              <div className={`${service.bgColor} p-3 rounded-lg`}><service.icon className={`h-6 w-6 ${service.color}`} /></div>
              <div className="flex-1">
                <h3 className="font-semibold text-gray-900 mb-2 group-hover:text-blue-600 transition-colors">{service.title}</h3>
                <p className="text-sm text-gray-600">{service.description}</p>
              </div>
              <ChevronRight className="h-5 w-5 text-gray-400 group-hover:translate-x-1 transition-transform"/>
            </CardContent>
          </Card>
        ))}
      </div>
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle className="text-lg font-semibold flex items-center">
              <Clock className="h-5 w-5 mr-2 text-blue-600" />Recent Tax Payments
            </CardTitle>
            {taxPayments.length > 0 && (
              <Button 
                variant="outline" 
                size="sm"
                onClick={handleExportHistory}
                disabled={isLoading}
              >
                <Download className="h-4 w-4 mr-2" />
                Export All
              </Button>
            )}
          </div>
        </CardHeader>
        <CardContent>
          {taxPayments.length === 0 ? (
            <div className="text-center py-8 text-gray-500"><Receipt className="h-12 w-12 mx-auto mb-4 text-gray-300" /><p className="font-medium">No tax payments yet.</p></div>
          ) : (
            <div className="space-y-3">
              {taxPayments.slice(0, 3).map(p => (
                <div key={p.id} className="flex items-center justify-between p-3 border rounded-lg">
                  <div className="flex items-center space-x-3">
                    <div className="bg-green-100 p-2 rounded-full"><CheckCircle className="h-4 w-4 text-green-600"/></div>
                    <div>
                      <p className="font-medium text-gray-900">{p.tax_type === 'DIRECT' ? 'Direct Tax' : p.tax_type === 'GST' ? 'GST Payment' : 'State Tax'}</p>
                      <p className="text-sm text-gray-500">{p.description}</p>
                    </div>
                  </div>
                  <div className="text-right">
                    <p className="font-semibold text-lg">₹{p.amount.toLocaleString('en-IN')}</p>
                    <p className="text-xs text-gray-400">{new Date(p.payment_date).toLocaleDateString('en-IN')}</p>
                  </div>
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
  
  const HistoryView = () => (
     <div className="space-y-6">
        <Button variant="ghost" onClick={() => setCurrentView('dashboard')}><ArrowLeft className="mr-2 h-4 w-4" /> Back to e-Tax Dashboard</Button>
        <Card>
            <CardHeader>
              <div className="flex items-center justify-between">
                <CardTitle>Complete Payment History</CardTitle>
                {taxPayments.length > 0 && (
                  <Button 
                    variant="outline" 
                    size="sm"
                    onClick={handleExportHistory}
                    disabled={isLoading}
                  >
                    <Download className="h-4 w-4 mr-2" />
                    Export History
                  </Button>
                )}
              </div>
            </CardHeader>
            <CardContent className="space-y-3">
            {taxPayments.length === 0 ? <p className="text-muted-foreground text-center py-4">No records to show.</p> : taxPayments.map(p => (
                 <div key={p.id} className="flex items-center justify-between p-3 border rounded-lg">
                  <div className="flex items-center space-x-3">
                    <div className="bg-green-100 p-2 rounded-full"><CheckCircle className="h-4 w-4 text-green-600"/></div>
                    <div>
                      <p className="font-medium text-gray-900">{p.tax_type === 'DIRECT' ? 'Direct Tax' : p.tax_type === 'GST' ? 'GST Payment' : 'State Tax'}</p>
                      <p className="text-sm text-gray-500">{p.description}</p>
                    </div>
                  </div>
                  <div className="flex items-center space-x-4">
                    <div className="text-right">
                      <p className="font-semibold text-lg">₹{p.amount.toLocaleString('en-IN')}</p>
                      <p className="text-xs text-gray-400">{new Date(p.payment_date).toLocaleDateString('en-IN')}</p>
                    </div>
                    <Button 
                      variant="outline" 
                      size="icon"
                      onClick={() => handleDownloadChallan(p.id)}
                      disabled={isLoading}
                    >
                      <Download className="h-4 w-4"/>
                    </Button>
                  </div>
                </div>
            ))}
            </CardContent>
        </Card>
    </div>
  );


  const renderView = () => {
    switch (currentView) {
      case 'direct': return <DirectTaxView onBack={() => setCurrentView('dashboard')} onSubmit={handlePaymentSubmit} isLoading={isLoading} />;
      case 'gst': return <GstView onBack={() => setCurrentView('dashboard')} onSubmit={handlePaymentSubmit} isLoading={isLoading} />;
      case 'state': return <StateTaxView onBack={() => setCurrentView('dashboard')} onSubmit={handlePaymentSubmit} isLoading={isLoading} />;
      case 'history': return <HistoryView />;
      case 'dashboard':
      default:
        return <DashboardView />;
    }
  };

  return (
    <BankingLayout>
      <div>
        {renderView()}
      </div>
    </BankingLayout>
  );
}