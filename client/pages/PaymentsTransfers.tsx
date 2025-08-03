import { BankingLayout } from "@/components/layout/BankingLayout";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { useAuth } from "@/contexts/AuthContext";
import { useNotifications } from "@/contexts/NotificationContext";
import { transfersApi, Beneficiary } from "@/services/transfersapi";
import { checkAnomalyAfterTransaction } from "@/services/anomalyService";
import { useState, useEffect, useCallback } from "react";
import { useNavigate } from "react-router-dom";
import {
  Send,
  UserPlus,
  Building,
  CreditCard,
  Smartphone,
  ArrowRightLeft,
  Clock,
  Shield,
  ChevronRight,
  Settings,
  X,
  Plus,
  Trash2,
} from "lucide-react";

export default function PaymentsTransfers() {
  const { user, fetchAndSetProfile } = useAuth();
  const { addNotification } = useNotifications();
  const navigate = useNavigate();
  const [beneficiaries, setBeneficiaries] = useState<Beneficiary[]>([]);
  const [recentTransfers, setRecentTransfers] = useState<any[]>([]);
  const [allTransfers, setAllTransfers] = useState<any[]>([]);
  const [showAddBeneficiary, setShowAddBeneficiary] = useState(false);
  const [showOwnTransfer, setShowOwnTransfer] = useState(false);
  const [showBeneficiaryTransfer, setShowBeneficiaryTransfer] = useState(false);
  const [showAllTransfers, setShowAllTransfers] = useState(false);
  const [selectedBeneficiary, setSelectedBeneficiary] = useState<Beneficiary | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  const [beneficiaryForm, setBeneficiaryForm] = useState({
    account_number: '',
    account_holder_name: '',
    ifsc_code: '',
    bank_name: '',
    nickname: '',
    is_within_securebank: true,
  });

  const [ownTransferForm, setOwnTransferForm] = useState({
    amount: '',
  });

  const [transferForm, setTransferForm] = useState({
    amount: '',
    remarks: '',
  });

  useEffect(() => {
    loadBeneficiaries();
    loadRecentTransfers();
  }, []);

  const loadBeneficiaries = async () => {
    try {
      const response = await transfersApi.getBeneficiaries();
      if (response.success && response.beneficiaries) {
        setBeneficiaries(response.beneficiaries);
      }
    } catch (error) {
      console.error('Error loading beneficiaries:', error);
    }
  };

  const loadRecentTransfers = async () => {
    try {
      const response = await transfersApi.getRecentTransfers();
      if (response.success && response.transfers) {
        setRecentTransfers(response.transfers.slice(0, 3));
        setAllTransfers(response.transfers);
      }
    } catch (error) {
      console.error('Error loading recent transfers:', error);
    }
  };

  const handleAddBeneficiary = async () => {
    console.log('Form data before validation:', beneficiaryForm);
    
    if (!beneficiaryForm.account_number.trim() || !beneficiaryForm.account_holder_name.trim() || !beneficiaryForm.nickname.trim()) {
      alert('Please fill all required fields: Account Number, Account Holder Name, and Nickname');
      return;
    }

    if (!beneficiaryForm.is_within_securebank && (!beneficiaryForm.ifsc_code.trim() || !beneficiaryForm.bank_name.trim())) {
      alert('IFSC Code and Bank Name are required for external bank transfers');
      return;
    }

    // For SecureBank transfers, auto-fill IFSC and bank name if empty
    const beneficiaryData = {
      ...beneficiaryForm,
      ifsc_code: beneficiaryForm.is_within_securebank && !beneficiaryForm.ifsc_code.trim() ? 'SECB0000001' : beneficiaryForm.ifsc_code,
      bank_name: beneficiaryForm.is_within_securebank && !beneficiaryForm.bank_name.trim() ? 'SecureBank' : beneficiaryForm.bank_name,
    };
    
    console.log('Final beneficiary data:', beneficiaryData);

    setIsLoading(true);
    try {
      const response = await transfersApi.addBeneficiary(beneficiaryData);
      console.log('Add beneficiary response:', response);
      
      if (response.success) {
        setBeneficiaryForm({
          account_number: '',
          account_holder_name: '',
          ifsc_code: '',
          bank_name: '',
          nickname: '',
          is_within_securebank: true,
        });
        setShowAddBeneficiary(false);
        await loadBeneficiaries();
        alert('Beneficiary added successfully!');
      } else {
        console.error('Add beneficiary failed:', response);
        alert(response.error || 'Failed to add beneficiary');
      }
    } catch (error) {
      console.error('Error adding beneficiary:', error);
      alert('Error adding beneficiary');
    } finally {
      setIsLoading(false);
    }
  };

  const handleDeleteBeneficiary = async (beneficiaryId: string) => {
    if (!confirm('Are you sure you want to delete this beneficiary?')) return;

    try {
      const response = await transfersApi.deleteBeneficiary(beneficiaryId);
      if (response.success) {
        await loadBeneficiaries();
        alert('Beneficiary deleted successfully!');
      } else {
        alert(response.error || 'Failed to delete beneficiary');
      }
    } catch (error) {
      console.error('Error deleting beneficiary:', error);
      alert('Error deleting beneficiary');
    }
  };

  const handleOwnTransfer = async () => {
    if (!ownTransferForm.amount || parseFloat(ownTransferForm.amount) <= 0) {
      alert('Please enter a valid amount');
      return;
    }

    setIsLoading(true);
    try {
      const response = await transfersApi.transferToOwnAccount(parseFloat(ownTransferForm.amount), addNotification);
      if (response.success) {
        setOwnTransferForm({ amount: '' });
        setShowOwnTransfer(false);
        await fetchAndSetProfile();
        await loadRecentTransfers();
        alert('Transfer successful!');
        
        // Check for anomalies after successful transfer
        const anomalyDetected = await checkAnomalyAfterTransaction(
          'own_account_transfer', 
          parseFloat(ownTransferForm.amount),
          navigate,
          '/transfers'
        );
        
        if (anomalyDetected) {
          // User will be redirected to anomaly verification page
          return;
        }
      } else {
        alert(response.error || 'Transfer failed');
      }
    } catch (error) {
      console.error('Error processing transfer:', error);
      alert('Error processing transfer');
    } finally {
      setIsLoading(false);
    }
  };

  const handleBeneficiaryTransfer = async () => {
    if (!selectedBeneficiary || !transferForm.amount || parseFloat(transferForm.amount) <= 0) {
      alert('Please select a beneficiary and enter a valid amount');
      return;
    }

    if (parseFloat(transferForm.amount) > (user?.balance || 0)) {
      alert('Insufficient balance');
      return;
    }

    setIsLoading(true);
    try {
      const response = await transfersApi.transferToBeneficiary({
        beneficiary_id: selectedBeneficiary.id,
        amount: parseFloat(transferForm.amount),
        remarks: transferForm.remarks,
      }, selectedBeneficiary.account_holder_name, addNotification);
      if (response.success) {
        setTransferForm({ amount: '', remarks: '' });
        setSelectedBeneficiary(null);
        setShowBeneficiaryTransfer(false);
        await fetchAndSetProfile();
        await loadRecentTransfers();
        alert('Transfer successful!');
        
        // Check for anomalies after successful transfer
        const anomalyDetected = await checkAnomalyAfterTransaction(
          'beneficiary_transfer', 
          parseFloat(transferForm.amount),
          navigate,
          '/transfers'
        );
        
        if (anomalyDetected) {
          // User will be redirected to anomaly verification page
          return;
        }
      } else {
        alert(response.error || 'Transfer failed');
      }
    } catch (error) {
      console.error('Error processing transfer:', error);
      alert('Error processing transfer');
    } finally {
      setIsLoading(false);
    }
  };
  const transferServices = [
    {
      title: "Add & Manage Beneficiary",
      description: "Add new beneficiaries and manage existing ones",
      icon: UserPlus,
      color: "text-blue-600",
      bgColor: "bg-blue-100",
      href: "/transfers/beneficiary",
    },
    {
      title: "Fund transfer (Own A/c)",
      description: "Add money to your own account for testing",
      icon: ArrowRightLeft,
      color: "text-green-600",
      bgColor: "bg-green-100",
      href: "/transfers/own-account",
    },
    {
      title: "Accounts of Others - Within SecureBank",
      description: "Transfer to other SecureBank account holders",
      icon: Building,
      color: "text-purple-600",
      bgColor: "bg-purple-100",
      href: "/transfers/within-securebank",
    },
  ];


  return (
    <BankingLayout>
      <div className="space-y-6 animate-slide-in">
        {/* Header */}
        <div className="bg-banking-gradient rounded-lg p-6 text-white">
          <div className="flex items-center">
            <Send className="h-8 w-8 mr-4" />
            <div>
              <h1 className="text-2xl font-bold">Payments / Transfers</h1>
              <p className="text-blue-100">
                Transfer money securely and instantly
              </p>
            </div>
          </div>
        </div>

        {/* Transfer Services Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {transferServices.map((service, index) => {
            const Icon = service.icon;
            return (
              <Card
                key={index}
                className="hover:shadow-lg transition-shadow cursor-pointer group"
                onClick={() => {
                  if (index === 0) setShowAddBeneficiary(true);
                  else if (index === 1) setShowOwnTransfer(true);
                  else if (index === 2) setShowBeneficiaryTransfer(true);
                }}
              >
                <CardContent className="p-6">
                  <div className="flex items-start space-x-4">
                    <div className={`${service.bgColor} p-3 rounded-lg`}>
                      <Icon className={`h-6 w-6 ${service.color}`} />
                    </div>
                    <div className="flex-1">
                      <h3 className="font-semibold text-gray-900 mb-2 group-hover:text-banking-blue-600 transition-colors">
                        {service.title}
                      </h3>
                      <p className="text-sm text-gray-600 mb-3">
                        {service.description}
                      </p>
                      <Button
                        variant="ghost"
                        size="sm"
                        className="p-0 h-auto text-banking-blue-600 hover:text-banking-blue-700"
                      >
                        Access Service
                        <ChevronRight className="h-4 w-4 ml-1" />
                      </Button>
                    </div>
                  </div>
                </CardContent>
              </Card>
            );
          })}
        </div>

        {/* Quick Actions */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <Card>
            <CardHeader>
              <CardTitle className="text-lg font-semibold flex items-center">
                <Clock className="h-5 w-5 mr-2 text-blue-600" />
                Recent Transfers
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {recentTransfers.map((transfer) => (
                  <div
                    key={transfer.id}
                    className="flex items-center justify-between p-3 border rounded-lg hover:bg-gray-50"
                  >
                    <div className="flex items-center space-x-3">
                      <div className="bg-green-100 p-2 rounded-lg">
                        <Send className="h-4 w-4 text-green-600" />
                      </div>
                      <div>
                        <p className="font-medium text-gray-900">
                          {transfer.counterparty || transfer.description}
                        </p>
                        <p className="text-sm text-gray-600">
                          {transfer.type} • {transfer.created_at ? new Date(transfer.created_at).toLocaleDateString('en-IN') : 'N/A'}
                        </p>
                      </div>
                    </div>
                    <div className="text-right">
                      <p className="font-semibold text-gray-900">
                        ₹{Math.abs(transfer.amount).toLocaleString("en-IN")}
                      </p>
                      <p className="text-xs text-green-600">
                        {transfer.status}
                      </p>
                    </div>
                  </div>
                ))}
                <Button 
                  variant="outline" 
                  className="w-full"
                  onClick={() => setShowAllTransfers(true)}
                >
                  View All Transfers
                </Button>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="text-lg font-semibold flex items-center">
                <Shield className="h-5 w-5 mr-2 text-green-600" />
                Security Features
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div className="flex items-start space-x-3">
                  <div className="bg-green-100 p-2 rounded-lg mt-1">
                    <Shield className="h-4 w-4 text-green-600" />
                  </div>
                  <div>
                    <h4 className="font-medium text-gray-900">
                      Two-Factor Authentication
                    </h4>
                    <p className="text-sm text-gray-600">
                      Every transaction requires OTP verification
                    </p>
                  </div>
                </div>

                <div className="flex items-start space-x-3">
                  <div className="bg-blue-100 p-2 rounded-lg mt-1">
                    <Clock className="h-4 w-4 text-blue-600" />
                  </div>
                  <div>
                    <h4 className="font-medium text-gray-900">
                      Transaction Limits
                    </h4>
                    <p className="text-sm text-gray-600">
                      Daily transfer limit: ₹2,00,000
                    </p>
                  </div>
                </div>

                <div className="flex items-start space-x-3">
                  <div className="bg-purple-100 p-2 rounded-lg mt-1">
                    <Send className="h-4 w-4 text-purple-600" />
                  </div>
                  <div>
                    <h4 className="font-medium text-gray-900">
                      Instant Notifications
                    </h4>
                    <p className="text-sm text-gray-600">
                      SMS and email alerts for all transactions
                    </p>
                  </div>
                </div>

                <div className="pt-2">
                  <Button variant="outline" className="w-full">
                    <Settings className="h-4 w-4 mr-2" />
                    Manage Security Settings
                  </Button>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>


        {/* View All Transfers Modal */}
        {showAllTransfers && (
          <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
            <Card className="w-full max-w-4xl mx-4 max-h-[80vh] overflow-hidden">
              <CardHeader className="pb-3">
                <div className="flex items-center justify-between">
                  <CardTitle>All Transfers</CardTitle>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => setShowAllTransfers(false)}
                  >
                    <X className="h-4 w-4" />
                  </Button>
                </div>
              </CardHeader>
              <CardContent className="overflow-y-auto max-h-[60vh]">
                <div className="space-y-3">
                  {allTransfers.length === 0 ? (
                    <p className="text-center py-8 text-gray-500">
                      No transfers found.
                    </p>
                  ) : (
                    allTransfers.map((transfer) => (
                      <div
                        key={transfer.id}
                        className="flex items-center justify-between p-4 border rounded-lg hover:bg-gray-50"
                      >
                        <div className="flex items-center space-x-4">
                          <div className="bg-green-100 p-2 rounded-lg">
                            <Send className="h-4 w-4 text-green-600" />
                          </div>
                          <div>
                            <p className="font-medium text-gray-900">
                              {transfer.counterparty || transfer.description}
                            </p>
                            <p className="text-sm text-gray-600">
                              {transfer.type} • {transfer.created_at ? new Date(transfer.created_at).toLocaleDateString('en-IN') : 'N/A'}
                            </p>
                            {transfer.reference && (
                              <p className="text-xs text-gray-500">
                                Ref: {transfer.reference}
                              </p>
                            )}
                          </div>
                        </div>
                        <div className="text-right">
                          <p className="font-semibold text-gray-900">
                            ₹{Math.abs(transfer.amount).toLocaleString("en-IN")}
                          </p>
                          <p className="text-xs text-green-600">
                            {transfer.status}
                          </p>
                        </div>
                      </div>
                    ))
                  )}
                </div>
              </CardContent>
            </Card>
          </div>
        )}

        {/* Add Beneficiary Modal */}
        {showAddBeneficiary && (
          <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
            <Card className="w-full max-w-md mx-4">
              <CardHeader>
                <div className="flex items-center justify-between">
                  <CardTitle>Add Beneficiary</CardTitle>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => setShowAddBeneficiary(false)}
                  >
                    <X className="h-4 w-4" />
                  </Button>
                </div>
              </CardHeader>
              <CardContent className="space-y-4">
                <div>
                  <Label htmlFor="nickname">Nickname</Label>
                  <Input
                    id="nickname"
                    value={beneficiaryForm.nickname}
                    onChange={(e) => setBeneficiaryForm({...beneficiaryForm, nickname: e.target.value})}
                    placeholder="Enter nickname"
                  />
                </div>
                <div>
                  <Label htmlFor="account_holder_name">Account Holder Name</Label>
                  <Input
                    id="account_holder_name"
                    value={beneficiaryForm.account_holder_name}
                    onChange={(e) => setBeneficiaryForm({...beneficiaryForm, account_holder_name: e.target.value})}
                    placeholder="Enter account holder name"
                  />
                </div>
                <div>
                  <Label htmlFor="account_number">Account Number</Label>
                  <Input
                    id="account_number"
                    value={beneficiaryForm.account_number}
                    onChange={(e) => setBeneficiaryForm({...beneficiaryForm, account_number: e.target.value})}
                    placeholder="Enter account number"
                  />
                </div>
                {!beneficiaryForm.is_within_securebank && (
                  <>
                    <div>
                      <Label htmlFor="ifsc_code">IFSC Code</Label>
                      <Input
                        id="ifsc_code"
                        value={beneficiaryForm.ifsc_code}
                        onChange={(e) => setBeneficiaryForm({...beneficiaryForm, ifsc_code: e.target.value})}
                        placeholder="Enter IFSC code"
                      />
                    </div>
                    <div>
                      <Label htmlFor="bank_name">Bank Name</Label>
                      <Input
                        id="bank_name"
                        value={beneficiaryForm.bank_name}
                        onChange={(e) => setBeneficiaryForm({...beneficiaryForm, bank_name: e.target.value})}
                        placeholder="Enter bank name"
                      />
                    </div>
                  </>
                )}
                <div className="flex items-center space-x-2">
                  <input
                    type="checkbox"
                    id="is_within_securebank"
                    checked={beneficiaryForm.is_within_securebank}
                    onChange={(e) => setBeneficiaryForm({...beneficiaryForm, is_within_securebank: e.target.checked})}
                  />
                  <Label htmlFor="is_within_securebank">Within SecureBank</Label>
                </div>
                <Button
                  onClick={handleAddBeneficiary}
                  disabled={isLoading}
                  className="w-full"
                >
                  {isLoading ? 'Adding...' : 'Add Beneficiary'}
                </Button>
              </CardContent>
            </Card>
          </div>
        )}

        {/* Own Account Transfer Modal */}
        {showOwnTransfer && (
          <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
            <Card className="w-full max-w-md mx-4">
              <CardHeader>
                <div className="flex items-center justify-between">
                  <CardTitle>Fund Transfer (Own Account)</CardTitle>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => setShowOwnTransfer(false)}
                  >
                    <X className="h-4 w-4" />
                  </Button>
                </div>
              </CardHeader>
              <CardContent className="space-y-4">
                <div>
                  <Label htmlFor="amount">Amount</Label>
                  <Input
                    id="amount"
                    type="number"
                    value={ownTransferForm.amount}
                    onChange={(e) => setOwnTransferForm({amount: e.target.value})}
                    placeholder="Enter amount"
                  />
                </div>
                <p className="text-sm text-gray-600">
                  Current Balance: ₹{user?.balance?.toLocaleString('en-IN') || '0'}
                </p>
                <Button
                  onClick={handleOwnTransfer}
                  disabled={isLoading}
                  className="w-full"
                >
                  {isLoading ? 'Processing...' : 'Add Money'}
                </Button>
              </CardContent>
            </Card>
          </div>
        )}

        {/* Beneficiary Transfer Modal */}
        {showBeneficiaryTransfer && (
          <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
            <Card className="w-full max-w-md mx-4">
              <CardHeader>
                <div className="flex items-center justify-between">
                  <CardTitle>Transfer to Beneficiary</CardTitle>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => {
                      setShowBeneficiaryTransfer(false);
                      setSelectedBeneficiary(null);
                    }}
                  >
                    <X className="h-4 w-4" />
                  </Button>
                </div>
              </CardHeader>
              <CardContent className="space-y-4">
                {!selectedBeneficiary ? (
                  <>
                    <div>
                      <Label>Select Beneficiary</Label>
                      <div className="space-y-2 mt-2 max-h-48 overflow-y-auto">
                        {beneficiaries.map((beneficiary) => (
                          <div
                            key={beneficiary.id}
                            className="p-3 border rounded-lg cursor-pointer hover:bg-gray-50 flex items-center justify-between"
                            onClick={() => setSelectedBeneficiary(beneficiary)}
                          >
                            <div>
                              <p className="font-medium">{beneficiary.nickname}</p>
                              <p className="text-sm text-gray-600">
                                {beneficiary.account_holder_name} • ****{beneficiary.account_number.slice(-4)}
                              </p>
                            </div>
                            <Button
                              variant="ghost"
                              size="sm"
                              onClick={(e) => {
                                e.stopPropagation();
                                handleDeleteBeneficiary(beneficiary.id);
                              }}
                            >
                              <Trash2 className="h-4 w-4" />
                            </Button>
                          </div>
                        ))}
                      </div>
                    </div>
                    {beneficiaries.length === 0 && (
                      <p className="text-sm text-gray-600 text-center py-4">
                        No beneficiaries found. Add one first.
                      </p>
                    )}
                  </>
                ) : (
                  <>
                    <div className="bg-gray-50 p-3 rounded-lg">
                      <p className="font-medium">{selectedBeneficiary.nickname}</p>
                      <p className="text-sm text-gray-600">
                        {selectedBeneficiary.account_holder_name} • ****{selectedBeneficiary.account_number.slice(-4)}
                      </p>
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => setSelectedBeneficiary(null)}
                        className="mt-2"
                      >
                        Change Beneficiary
                      </Button>
                    </div>
                    <div>
                      <Label htmlFor="transfer_amount">Amount</Label>
                      <Input
                        id="transfer_amount"
                        type="number"
                        value={transferForm.amount}
                        onChange={(e) => setTransferForm({...transferForm, amount: e.target.value})}
                        placeholder="Enter amount"
                      />
                    </div>
                    <div>
                      <Label htmlFor="remarks">Remarks (Optional)</Label>
                      <Textarea
                        id="remarks"
                        value={transferForm.remarks}
                        onChange={(e) => setTransferForm({...transferForm, remarks: e.target.value})}
                        placeholder="Enter remarks"
                      />
                    </div>
                    <p className="text-sm text-gray-600">
                      Available Balance: ₹{user?.balance?.toLocaleString('en-IN') || '0'}
                    </p>
                    <Button
                      onClick={handleBeneficiaryTransfer}
                      disabled={isLoading}
                      className="w-full"
                    >
                      {isLoading ? 'Processing...' : 'Transfer Money'}
                    </Button>
                  </>
                )}
              </CardContent>
            </Card>
          </div>
        )}
      </div>
    </BankingLayout>
  );
}
