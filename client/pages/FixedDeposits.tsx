import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { BankingLayout } from "@/components/layout/BankingLayout";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { useToast } from "@/hooks/use-toast";
import {
  PiggyBank,
  Calculator,
  TrendingUp,
  Calendar,
  DollarSign,
  Plus,
  Eye,
  Download,
  RefreshCw,
  AlertCircle,
  CheckCircle,
  Clock,
  Banknote,
  FileText,
} from "lucide-react";
import { fixedDepositsApi, FixedDeposit, FDDetails, FDCertificate } from "@/services/api";
import { checkAnomalyAfterTransaction } from "@/services/anomalyService";

export default function FixedDeposits() {
  const [fdAmount, setFdAmount] = useState("");
  const [fdTenure, setFdTenure] = useState("");
  const [fdType, setFdType] = useState("REGULAR");
  const [calculatedMaturity, setCalculatedMaturity] = useState(0);
  const [isLoading, setIsLoading] = useState(false);
  const [fixedDeposits, setFixedDeposits] = useState<FixedDeposit[]>([]);
  const [selectedFD, setSelectedFD] = useState<FDDetails | null>(null);
  const [certificateData, setCertificateData] = useState<FDCertificate | null>(null);
  const [showDetailsDialog, setShowDetailsDialog] = useState(false);
  const [showCertificateDialog, setShowCertificateDialog] = useState(false);
  const { toast } = useToast();
  const navigate = useNavigate();

  // Interest rates configuration
  const interestRates = {
    REGULAR: { 12: 6.8, 24: 7.5, 36: 7.2, 60: 7.8, 120: 8.1 },
    SENIOR: { 12: 7.3, 24: 8.0, 36: 7.7, 60: 8.3, 120: 8.6 },
    TAX_SAVING: { 60: 7.2 }
  };

  // Load fixed deposits on component mount
  useEffect(() => {
    loadFixedDeposits();
  }, []);

  const loadFixedDeposits = async () => {
    try {
      setIsLoading(true);
      const response = await fixedDepositsApi.getFixedDeposits() as any;
      if (response.success) {
        setFixedDeposits(response.fixedDeposits);
      }
    } catch (error) {
      toast({
        title: "Error",
        description: "Failed to load fixed deposits",
        variant: "destructive",
      });
    } finally {
      setIsLoading(false);
    }
  };

  const calculateMaturity = () => {
    if (!fdAmount || !fdTenure) return;

    const principal = parseFloat(fdAmount);
    const months = parseInt(fdTenure);
    
    // Get appropriate rate based on type and tenure
    const rates = interestRates[fdType as keyof typeof interestRates];
    const rate = rates[months as keyof typeof rates] || 7.5;

    // Compound interest calculation (quarterly compounding)
    const quarters = months / 3;
    const maturityAmount = principal * Math.pow(1 + rate / 400, quarters);
    setCalculatedMaturity(maturityAmount);
  };

  const createFixedDeposit = async () => {
    if (!fdAmount || !fdTenure || !fdType) {
      toast({
        title: "Error",
        description: "Please fill in all required fields",
        variant: "destructive",
      });
      return;
    }

    try {
      setIsLoading(true);
      const response = await fixedDepositsApi.createFixedDeposit(
        parseFloat(fdAmount),
        parseInt(fdTenure),
        fdType
      ) as any;

      if (response.success) {
        toast({
          title: "Success",
          description: "Fixed Deposit created successfully!",
        });
        
        // Reset form
        setFdAmount("");
        setFdTenure("");
        setFdType("REGULAR");
        setCalculatedMaturity(0);
        
        // Reload fixed deposits
        await loadFixedDeposits();
        
        // Check for anomalies after successful FD creation
        const anomalyDetected = await checkAnomalyAfterTransaction(
          'fd_created',
          parseFloat(fdAmount),
          navigate,
          '/fixed-deposits'
        );
        
        if (anomalyDetected) {
          // User will be redirected to anomaly verification page
          return;
        }
      }
    } catch (error: any) {
      toast({
        title: "Error",
        description: error.message || "Failed to create fixed deposit",
        variant: "destructive",
      });
    } finally {
      setIsLoading(false);
    }
  };

  const handleFDDetails = async (fdId: string) => {
    try {
      const response = await fixedDepositsApi.getFDDetails(fdId) as any;
      if (response.success) {
        setSelectedFD(response.fdDetails);
        setShowDetailsDialog(true);
      }
    } catch (error: any) {
      toast({
        title: "Error",
        description: error.message || "Failed to load FD details",
        variant: "destructive",
      });
    }
  };

  const handleFDCertificate = async (fdId: string) => {
    try {
      const response = await fixedDepositsApi.getFDCertificate(fdId) as any;
      if (response.success) {
        setCertificateData(response.certificate);
        setShowCertificateDialog(true);
      }
    } catch (error: any) {
      toast({
        title: "Error",
        description: error.message || "Failed to load FD certificate",
        variant: "destructive",
      });
    }
  };

  const exportToPDF = async () => {
    try {
      setIsLoading(true);
      const response = await fixedDepositsApi.exportFixedDeposits() as any;
      if (response.success) {
        // Generate PDF content
        const pdfContent = generatePDFContent(response.exportData);
        
        // Create and download PDF
        const blob = new Blob([pdfContent], { type: 'application/pdf' });
        const url = window.URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = `fixed-deposits-${new Date().toISOString().split('T')[0]}.pdf`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        window.URL.revokeObjectURL(url);

        toast({
          title: "Success",
          description: "Fixed deposits exported successfully!",
        });
      }
    } catch (error: any) {
      toast({
        title: "Error",
        description: error.message || "Failed to export fixed deposits",
        variant: "destructive",
      });
    } finally {
      setIsLoading(false);
    }
  };

  const generatePDFContent = (data: any) => {
    // Simple PDF content generation (in a real app, you'd use a proper PDF library)
    const content = `
Fixed Deposits Report
Generated on: ${data.exportDate}

Customer: ${data.customerName}
Account: ${data.accountNumber}

Summary:
- Total Investment: ₹${data.summary.totalInvestment.toLocaleString('en-IN')}
- Interest Earned: ₹${data.summary.totalInterestEarned.toLocaleString('en-IN')}
- Active FDs: ${data.summary.activeFDs}
- Total FDs: ${data.summary.totalFDs}

Fixed Deposits:
${data.fixedDeposits.map(fd => `
FD ID: ${fd.id}
Amount: ₹${fd.amount.toLocaleString('en-IN')}
Interest Rate: ${fd.interestRate}%
Tenure: ${fd.tenure} months
Type: ${fd.type}
Status: ${fd.status}
Maturity Amount: ₹${fd.maturityAmount.toLocaleString('en-IN')}
`).join('\n')}
    `;
    return content;
  };

  // Calculate summary statistics
  const totalFDValue = fixedDeposits
    .filter((fd) => fd.status === "ACTIVE")
    .reduce((sum, fd) => sum + fd.amount, 0);

  const totalInterestEarned = fixedDeposits
    .filter((fd) => fd.status === "ACTIVE")
    .reduce((sum, fd) => sum + fd.interestEarned, 0);

  const activeFDs = fixedDeposits.filter((fd) => fd.status === "ACTIVE");
  const maturedFDs = fixedDeposits.filter((fd) => fd.status === "MATURED");

  return (
    <BankingLayout>
      <div className="space-y-6 animate-slide-in">
        {/* Header */}
        <div className="bg-banking-gradient rounded-lg p-6 text-white">
          <div className="flex items-center justify-between">
            <div className="flex items-center">
              <PiggyBank className="h-8 w-8 mr-4" />
              <div>
                <h1 className="text-2xl font-bold">Fixed Deposits</h1>
                <p className="text-blue-100">
                  Secure your future with guaranteed returns
                </p>
              </div>
            </div>
            <div className="text-right">
              <p className="text-sm text-blue-100">Total FD Value</p>
              <p className="text-2xl font-bold">
                ₹{totalFDValue.toLocaleString("en-IN")}
              </p>
            </div>
          </div>
        </div>

        {/* Overview Cards */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
          <Card className="bg-gradient-to-r from-green-500 to-green-600 text-white">
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-green-100 text-sm">Active FDs</p>
                  <p className="text-2xl font-bold">{activeFDs.length}</p>
                </div>
                <PiggyBank className="h-8 w-8 text-green-100" />
              </div>
            </CardContent>
          </Card>

          <Card className="bg-gradient-to-r from-blue-500 to-blue-600 text-white">
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-blue-100 text-sm">Total Investment</p>
                  <p className="text-2xl font-bold">
                    ₹{totalFDValue.toLocaleString("en-IN")}
                  </p>
                </div>
                <Banknote className="h-8 w-8 text-blue-100" />
              </div>
            </CardContent>
          </Card>

          <Card className="bg-gradient-to-r from-purple-500 to-purple-600 text-white">
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-purple-100 text-sm">Interest Earned</p>
                  <p className="text-2xl font-bold">
                    ₹{totalInterestEarned.toLocaleString("en-IN")}
                  </p>
                </div>
                <TrendingUp className="h-8 w-8 text-purple-100" />
              </div>
            </CardContent>
          </Card>

          <Card className="bg-gradient-to-r from-orange-500 to-orange-600 text-white">
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-orange-100 text-sm">Average Rate</p>
                  <p className="text-2xl font-bold">7.5%</p>
                </div>
                <Calculator className="h-8 w-8 text-orange-100" />
              </div>
            </CardContent>
          </Card>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Create New FD */}
          <Card>
            <CardHeader>
              <CardTitle className="text-lg font-semibold flex items-center">
                <Plus className="h-5 w-5 mr-2 text-green-600" />
                Open New Fixed Deposit
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-6">
                <div>
                  <Label htmlFor="amount" className="text-sm font-medium">
                    Deposit Amount (₹)
                  </Label>
                  <Input
                    id="amount"
                    type="number"
                    value={fdAmount}
                    onChange={(e) => setFdAmount(e.target.value)}
                    placeholder="Minimum ₹1,000"
                    className="mt-1"
                  />
                  <p className="text-xs text-gray-500 mt-1">
                    Minimum: ₹1,000 | Maximum: ₹2,00,00,000
                  </p>
                </div>

                <div>
                  <Label className="text-sm font-medium">
                    Fixed Deposit Type
                  </Label>
                  <RadioGroup
                    value={fdType}
                    onValueChange={setFdType}
                    className="mt-2"
                  >
                    <div className="flex items-center space-x-2">
                      <RadioGroupItem value="REGULAR" id="regular" />
                      <Label htmlFor="regular">Regular FD</Label>
                    </div>
                    <div className="flex items-center space-x-2">
                      <RadioGroupItem value="SENIOR" id="senior" />
                      <Label htmlFor="senior">Senior Citizen FD</Label>
                    </div>
                    <div className="flex items-center space-x-2">
                      <RadioGroupItem value="TAX_SAVING" id="tax" />
                      <Label htmlFor="tax">Tax Saving FD (5 years)</Label>
                    </div>
                  </RadioGroup>
                </div>

                <div>
                  <Label htmlFor="tenure" className="text-sm font-medium">
                    Tenure (Months)
                  </Label>
                  <Select value={fdTenure} onValueChange={setFdTenure}>
                    <SelectTrigger className="mt-1">
                      <SelectValue placeholder="Select tenure" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="12">12 Months (6.8%)</SelectItem>
                      <SelectItem value="24">24 Months (7.5%)</SelectItem>
                      <SelectItem value="36">36 Months (7.2%)</SelectItem>
                      <SelectItem value="60">60 Months (7.8%)</SelectItem>
                      <SelectItem value="120">120 Months (8.1%)</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div className="flex space-x-4">
                  <Button onClick={calculateMaturity} className="flex-1">
                    <Calculator className="h-4 w-4 mr-2" />
                    Calculate
                  </Button>
                  <Button 
                    onClick={createFixedDeposit} 
                    className="flex-1"
                    disabled={isLoading}
                  >
                    {isLoading ? (
                      <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
                    ) : (
                      <Plus className="h-4 w-4 mr-2" />
                    )}
                    Open FD
                  </Button>
                </div>

                {calculatedMaturity > 0 && (
                  <div className="mt-4 p-4 bg-green-50 rounded-lg border border-green-200">
                    <h4 className="font-semibold text-green-900 mb-2">
                      Maturity Calculation
                    </h4>
                    <div className="space-y-1 text-sm">
                      <div className="flex justify-between">
                        <span>Principal Amount:</span>
                        <span className="font-medium">
                          ₹{parseFloat(fdAmount).toLocaleString("en-IN")}
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span>Maturity Amount:</span>
                        <span className="font-medium text-green-600">
                          ₹{calculatedMaturity.toLocaleString("en-IN")}
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span>Interest Earned:</span>
                        <span className="font-medium text-green-600">
                          ₹{(calculatedMaturity - parseFloat(fdAmount)).toLocaleString("en-IN")}
                        </span>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </CardContent>
          </Card>

          {/* Interest Rates */}
          <Card>
            <CardHeader>
              <CardTitle className="text-lg font-semibold flex items-center">
                <TrendingUp className="h-5 w-5 mr-2 text-blue-600" />
                Current Interest Rates
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b">
                      <th className="text-left py-2 font-medium">Tenure</th>
                      <th className="text-right py-2 font-medium">Regular</th>
                      <th className="text-right py-2 font-medium">Senior</th>
                      <th className="text-right py-2 font-medium">Tax Saving</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(interestRates.REGULAR).map(([tenure, rate]) => (
                      <tr key={tenure} className="border-b hover:bg-gray-50">
                        <td className="py-2 font-medium">{tenure} months</td>
                        <td className="py-2 text-right">{rate}%</td>
                        <td className="py-2 text-right text-green-600">
                          {interestRates.SENIOR[parseInt(tenure)]}%
                        </td>
                        <td className="py-2 text-right">
                          {interestRates.TAX_SAVING[parseInt(tenure)] ? `${interestRates.TAX_SAVING[parseInt(tenure)]}%` : "-"}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              <div className="mt-4 p-3 bg-blue-50 rounded-lg">
                <p className="text-xs text-blue-700">
                  <AlertCircle className="h-3 w-3 inline mr-1" />
                  Interest rates are subject to change. Senior citizen rates applicable for age 60+.
                </p>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Existing FDs */}
        <Card>
          <CardHeader>
            <CardTitle className="text-lg font-semibold flex items-center justify-between">
              <div className="flex items-center">
                <Eye className="h-5 w-5 mr-2 text-purple-600" />
                My Fixed Deposits
              </div>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <Tabs defaultValue="active" className="w-full">
              <TabsList>
                <TabsTrigger value="active">Active FDs ({activeFDs.length})</TabsTrigger>
                <TabsTrigger value="matured">Matured FDs ({maturedFDs.length})</TabsTrigger>
                <TabsTrigger value="all">All FDs ({fixedDeposits.length})</TabsTrigger>
              </TabsList>

              <TabsContent value="active" className="mt-4">
                <div className="space-y-4">
                  {activeFDs.length === 0 ? (
                    <div className="text-center py-8 text-gray-500">
                      <PiggyBank className="h-12 w-12 mx-auto mb-4 text-gray-300" />
                      <p>No active fixed deposits found.</p>
                    </div>
                  ) : (
                    activeFDs.map((fd) => (
                      <div
                        key={fd.id}
                        className="border rounded-lg p-6 hover:bg-gray-50 transition-colors"
                      >
                        <div className="flex justify-between items-start mb-4">
                          <div>
                            <h3 className="font-semibold text-gray-900">
                              {fd.type} FD
                            </h3>
                            <p className="text-sm text-gray-600">
                              FD ID: {fd.id}
                            </p>
                            <p className="text-xs text-gray-500">
                              Created: {new Date(fd.startDate).toLocaleDateString()}
                            </p>
                          </div>
                          <Badge
                            variant="default"
                            className="bg-green-100 text-green-800"
                          >
                            <CheckCircle className="h-3 w-3 mr-1" />
                            {fd.status}
                          </Badge>
                        </div>

                        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
                          <div>
                            <p className="text-xs text-gray-600">
                              Principal Amount
                            </p>
                            <p className="font-semibold text-lg">
                              ₹{fd.amount.toLocaleString("en-IN")}
                            </p>
                          </div>
                          <div>
                            <p className="text-xs text-gray-600">
                              Interest Rate
                            </p>
                            <p className="font-semibold text-lg text-green-600">
                              {fd.interestRate}%
                            </p>
                          </div>
                          <div>
                            <p className="text-xs text-gray-600">
                              Maturity Date
                            </p>
                            <p className="font-semibold">
                              {new Date(fd.maturityDate).toLocaleDateString()}
                            </p>
                          </div>
                          <div>
                            <p className="text-xs text-gray-600">
                              Maturity Amount
                            </p>
                            <p className="font-semibold text-lg text-blue-600">
                              ₹{fd.maturityAmount.toLocaleString("en-IN")}
                            </p>
                          </div>
                        </div>

                        <div className="flex justify-between items-center">
                          <div className="text-sm">
                            <span className="text-gray-600">
                              Interest Earned:{" "}
                            </span>
                            <span className="font-semibold text-green-600">
                              ₹{fd.interestEarned.toLocaleString("en-IN")}
                            </span>
                          </div>
                          <div className="flex space-x-2">
                            <Button 
                              variant="outline" 
                              size="sm"
                              onClick={() => handleFDDetails(fd.id)}
                            >
                              <Calendar className="h-4 w-4 mr-1" />
                              Details
                            </Button>
                            <Button 
                              variant="outline" 
                              size="sm"
                              onClick={() => handleFDCertificate(fd.id)}
                            >
                              <Download className="h-4 w-4 mr-1" />
                              Certificate
                            </Button>
                          </div>
                        </div>
                      </div>
                    ))
                  )}
                </div>
              </TabsContent>

              <TabsContent value="matured" className="mt-4">
                <div className="space-y-4">
                  {maturedFDs.length === 0 ? (
                    <div className="text-center py-8 text-gray-500">
                      <Clock className="h-12 w-12 mx-auto mb-4 text-gray-300" />
                      <p>No matured fixed deposits found.</p>
                    </div>
                  ) : (
                    maturedFDs.map((fd) => (
                      <div
                        key={fd.id}
                        className="border rounded-lg p-6 bg-gray-50"
                      >
                        <div className="flex justify-between items-start mb-4">
                          <div>
                            <h3 className="font-semibold text-gray-900">
                              {fd.type} FD
                            </h3>
                            <p className="text-sm text-gray-600">
                              FD ID: {fd.id}
                            </p>
                          </div>
                          <Badge variant="secondary">
                            <Clock className="h-3 w-3 mr-1" />
                            {fd.status}
                          </Badge>
                        </div>

                        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                          <div>
                            <p className="text-xs text-gray-600">Principal</p>
                            <p className="font-semibold">
                              ₹{fd.amount.toLocaleString("en-IN")}
                            </p>
                          </div>
                          <div>
                            <p className="text-xs text-gray-600">
                              Interest Rate
                            </p>
                            <p className="font-semibold">{fd.interestRate}%</p>
                          </div>
                          <div>
                            <p className="text-xs text-gray-600">Matured On</p>
                            <p className="font-semibold">
                              {new Date(fd.maturityDate).toLocaleDateString()}
                            </p>
                          </div>
                          <div>
                            <p className="text-xs text-gray-600">
                              Final Amount
                            </p>
                            <p className="font-semibold text-green-600">
                              ₹{fd.maturityAmount.toLocaleString("en-IN")}
                            </p>
                          </div>
                        </div>
                      </div>
                    ))
                  )}
                </div>
              </TabsContent>

              <TabsContent value="all" className="mt-4">
                <div className="space-y-4">
                  {fixedDeposits.length === 0 ? (
                    <div className="text-center py-8 text-gray-500">
                      <PiggyBank className="h-12 w-12 mx-auto mb-4 text-gray-300" />
                      <p>No fixed deposits found.</p>
                    </div>
                  ) : (
                    fixedDeposits.map((fd) => (
                      <div
                        key={fd.id}
                        className="border rounded-lg p-6 hover:bg-gray-50 transition-colors"
                      >
                        <div className="flex justify-between items-start mb-4">
                          <div>
                            <h3 className="font-semibold text-gray-900">
                              {fd.type} FD
                            </h3>
                            <p className="text-sm text-gray-600">
                              FD ID: {fd.id}
                            </p>
                            <p className="text-xs text-gray-500">
                              Created: {new Date(fd.startDate).toLocaleDateString()}
                            </p>
                          </div>
                          <Badge
                            variant={fd.status === "ACTIVE" ? "default" : "secondary"}
                            className={fd.status === "ACTIVE" ? "bg-green-100 text-green-800" : ""}
                          >
                            {fd.status === "ACTIVE" ? (
                              <CheckCircle className="h-3 w-3 mr-1" />
                            ) : (
                              <Clock className="h-3 w-3 mr-1" />
                            )}
                            {fd.status}
                          </Badge>
                        </div>

                        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                          <div>
                            <p className="text-xs text-gray-600">Principal</p>
                            <p className="font-semibold">
                              ₹{fd.amount.toLocaleString("en-IN")}
                            </p>
                          </div>
                          <div>
                            <p className="text-xs text-gray-600">
                              Interest Rate
                            </p>
                            <p className="font-semibold">{fd.interestRate}%</p>
                          </div>
                          <div>
                            <p className="text-xs text-gray-600">Maturity Date</p>
                            <p className="font-semibold">
                              {new Date(fd.maturityDate).toLocaleDateString()}
                            </p>
                          </div>
                          <div>
                            <p className="text-xs text-gray-600">
                              Maturity Amount
                            </p>
                            <p className="font-semibold text-blue-600">
                              ₹{fd.maturityAmount.toLocaleString("en-IN")}
                            </p>
                          </div>
                        </div>
                      </div>
                    ))
                  )}
                </div>
              </TabsContent>
            </Tabs>
          </CardContent>
        </Card>

        {/* FD Details Dialog */}
        <Dialog open={showDetailsDialog} onOpenChange={setShowDetailsDialog}>
          <DialogContent className="max-w-2xl">
            <DialogHeader>
              <DialogTitle className="flex items-center">
                <FileText className="h-5 w-5 mr-2" />
                Fixed Deposit Details
              </DialogTitle>
            </DialogHeader>
            {selectedFD && (
              <div className="space-y-6">
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <Label className="text-sm font-medium text-gray-600">FD ID</Label>
                    <p className="font-semibold">{selectedFD.id}</p>
                  </div>
                  <div>
                    <Label className="text-sm font-medium text-gray-600">Customer Name</Label>
                    <p className="font-semibold">{selectedFD.customerName}</p>
                  </div>
                  <div>
                    <Label className="text-sm font-medium text-gray-600">Account Number</Label>
                    <p className="font-semibold">{selectedFD.accountNumber}</p>
                  </div>
                  <div>
                    <Label className="text-sm font-medium text-gray-600">FD Type</Label>
                    <p className="font-semibold">{selectedFD.type}</p>
                  </div>
                  <div>
                    <Label className="text-sm font-medium text-gray-600">Principal Amount</Label>
                    <p className="font-semibold">₹{selectedFD.amount.toLocaleString("en-IN")}</p>
                  </div>
                  <div>
                    <Label className="text-sm font-medium text-gray-600">Interest Rate</Label>
                    <p className="font-semibold text-green-600">{selectedFD.interestRate}%</p>
                  </div>
                  <div>
                    <Label className="text-sm font-medium text-gray-600">Tenure</Label>
                    <p className="font-semibold">{selectedFD.tenure} months</p>
                  </div>
                  <div>
                    <Label className="text-sm font-medium text-gray-600">Status</Label>
                    <Badge variant={selectedFD.status === "ACTIVE" ? "default" : "secondary"}>
                      {selectedFD.status}
                    </Badge>
                  </div>
                  <div>
                    <Label className="text-sm font-medium text-gray-600">Start Date</Label>
                    <p className="font-semibold">{new Date(selectedFD.startDate).toLocaleDateString()}</p>
                  </div>
                  <div>
                    <Label className="text-sm font-medium text-gray-600">Maturity Date</Label>
                    <p className="font-semibold">{new Date(selectedFD.maturityDate).toLocaleDateString()}</p>
                  </div>
                  <div>
                    <Label className="text-sm font-medium text-gray-600">Interest Earned</Label>
                    <p className="font-semibold text-green-600">₹{selectedFD.interestEarned.toLocaleString("en-IN")}</p>
                  </div>
                  <div>
                    <Label className="text-sm font-medium text-gray-600">Maturity Amount</Label>
                    <p className="font-semibold text-blue-600">₹{selectedFD.maturityAmount.toLocaleString("en-IN")}</p>
                  </div>
                </div>
              </div>
            )}
          </DialogContent>
        </Dialog>

        {/* FD Certificate Dialog */}
        <Dialog open={showCertificateDialog} onOpenChange={setShowCertificateDialog}>
          <DialogContent className="max-w-2xl">
            <DialogHeader>
              <DialogTitle className="flex items-center">
                <FileText className="h-5 w-5 mr-2" />
                Fixed Deposit Certificate
              </DialogTitle>
            </DialogHeader>
            {certificateData && (
              <div className="space-y-6">
                <div className="border-2 border-gray-200 rounded-lg p-6 bg-gradient-to-br from-blue-50 to-indigo-50">
                  <div className="text-center mb-6">
                    <h2 className="text-2xl font-bold text-gray-800">{certificateData.bankName}</h2>
                    <p className="text-gray-600">Fixed Deposit Certificate</p>
                  </div>
                  
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <Label className="text-sm font-medium text-gray-600">Certificate Number</Label>
                      <p className="font-semibold">{certificateData.certificateNumber}</p>
                    </div>
                    <div>
                      <Label className="text-sm font-medium text-gray-600">Issue Date</Label>
                      <p className="font-semibold">{certificateData.issueDate}</p>
                    </div>
                    <div>
                      <Label className="text-sm font-medium text-gray-600">Customer Name</Label>
                      <p className="font-semibold">{certificateData.customerName}</p>
                    </div>
                    <div>
                      <Label className="text-sm font-medium text-gray-600">Account Number</Label>
                      <p className="font-semibold">{certificateData.accountNumber}</p>
                    </div>
                    <div>
                      <Label className="text-sm font-medium text-gray-600">Phone</Label>
                      <p className="font-semibold">{certificateData.phone}</p>
                    </div>
                    <div>
                      <Label className="text-sm font-medium text-gray-600">Branch</Label>
                      <p className="font-semibold">{certificateData.branch}</p>
                    </div>
                    <div>
                      <Label className="text-sm font-medium text-gray-600">FD Type</Label>
                      <p className="font-semibold">{certificateData.type}</p>
                    </div>
                    <div>
                      <Label className="text-sm font-medium text-gray-600">Principal Amount</Label>
                      <p className="font-semibold">₹{certificateData.amount.toLocaleString("en-IN")}</p>
                    </div>
                    <div>
                      <Label className="text-sm font-medium text-gray-600">Interest Rate</Label>
                      <p className="font-semibold text-green-600">{certificateData.interestRate}%</p>
                    </div>
                    <div>
                      <Label className="text-sm font-medium text-gray-600">Tenure</Label>
                      <p className="font-semibold">{certificateData.tenure} months</p>
                    </div>
                    <div>
                      <Label className="text-sm font-medium text-gray-600">Start Date</Label>
                      <p className="font-semibold">{new Date(certificateData.startDate).toLocaleDateString()}</p>
                    </div>
                    <div>
                      <Label className="text-sm font-medium text-gray-600">Maturity Date</Label>
                      <p className="font-semibold">{new Date(certificateData.maturityDate).toLocaleDateString()}</p>
                    </div>
                    <div>
                      <Label className="text-sm font-medium text-gray-600">Maturity Amount</Label>
                      <p className="font-semibold text-blue-600">₹{certificateData.maturityAmount.toLocaleString("en-IN")}</p>
                    </div>
                  </div>
                  
                  <div className="mt-6 text-center">
                    <p className="text-xs text-gray-500">
                      This is a digital certificate. For official purposes, please contact your branch.
                    </p>
                  </div>
                </div>
              </div>
            )}
          </DialogContent>
        </Dialog>
      </div>
    </BankingLayout>
  );
}
