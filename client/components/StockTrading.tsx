import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { stocksApi, Stock } from "@/services/api";
import { useAuth } from "@/contexts/AuthContext";
import { checkAnomalyAfterTransaction } from "@/services/anomalyService";
import {
  ShoppingCart,
  DollarSign,
  Loader2,
  AlertCircle,
  CheckCircle,
  TrendingUp,
  TrendingDown,
} from "lucide-react";

interface StockTradingProps {
  stock: Stock;
  onTransactionComplete?: () => void;
}

export function StockTrading({
  stock,
  onTransactionComplete,
}: StockTradingProps) {
  const [quantity, setQuantity] = useState("");
  const [isTrading, setIsTrading] = useState(false);
  const [message, setMessage] = useState<{
    type: "success" | "error";
    text: string;
  } | null>(null);

  const { user } = useAuth();
  const navigate = useNavigate();

  const handleBuy = async () => {
    if (!quantity || parseInt(quantity) <= 0) {
      setMessage({ type: "error", text: "Please enter a valid quantity" });
      return;
    }

    const shares = parseInt(quantity);
    const totalCost = shares * stock.price * 1.005; // Including 0.5% brokerage

    if (user && user.balance < totalCost) {
      setMessage({ type: "error", text: "Insufficient balance" });
      return;
    }

    try {
      setIsTrading(true);
      setMessage(null);

      const result = await stocksApi.buyStock(stock.symbol, shares);

      if (result.success) {
        setMessage({
          type: "success",
          text: `Successfully bought ${shares} shares of ${stock.symbol} for ₹${totalCost.toLocaleString("en-IN")}`,
        });
        setQuantity("");
        onTransactionComplete?.();
        
        // Check for anomalies after successful stock purchase
        const anomalyDetected = await checkAnomalyAfterTransaction(
          'stock_buy',
          totalCost,
          navigate,
          `/stocks/${stock.symbol}`
        );
        
        if (anomalyDetected) {
          // User will be redirected to anomaly verification page
          return;
        }
      } else {
        setMessage({
          type: "error",
          text: result.message || "Failed to buy stock",
        });
      }
    } catch (error: any) {
      setMessage({
        type: "error",
        text: error.message || "Failed to buy stock",
      });
    } finally {
      setIsTrading(false);
    }
  };

  const handleSell = async () => {
    if (!quantity || parseInt(quantity) <= 0) {
      setMessage({ type: "error", text: "Please enter a valid quantity" });
      return;
    }

    const shares = parseInt(quantity);

    try {
      setIsTrading(true);
      setMessage(null);

      const result = await stocksApi.sellStock(stock.symbol, shares);

      if (result.success) {
        const netProceeds = shares * stock.price * 0.995; // After 0.5% brokerage
        setMessage({
          type: "success",
          text: `Successfully sold ${shares} shares of ${stock.symbol} for ₹${netProceeds.toLocaleString("en-IN")}`,
        });
        setQuantity("");
        onTransactionComplete?.();
        
        // Check for anomalies after successful stock sale
        const anomalyDetected = await checkAnomalyAfterTransaction(
          'stock_sell',
          netProceeds,
          navigate,
          `/stocks/${stock.symbol}`
        );
        
        if (anomalyDetected) {
          // User will be redirected to anomaly verification page
          return;
        }
      } else {
        setMessage({
          type: "error",
          text: result.message || "Failed to sell stock",
        });
      }
    } catch (error: any) {
      setMessage({
        type: "error",
        text: error.message || "Failed to sell stock",
      });
    } finally {
      setIsTrading(false);
    }
  };

  const totalCost = quantity ? parseFloat(quantity) * stock.price * 1.005 : 0;
  const netProceeds = quantity ? parseFloat(quantity) * stock.price * 0.995 : 0;

  return (
    <Card className="w-full max-w-md">
      <CardHeader>
        <CardTitle className="flex items-center justify-between">
          <div>
            <h3 className="text-lg font-semibold">{stock.symbol}</h3>
            <p className="text-sm text-gray-600">{stock.name}</p>
          </div>
          <div className="text-right">
            <p className="text-xl font-bold">₹{stock.price.toFixed(2)}</p>
            <Badge
              variant={stock.change >= 0 ? "default" : "destructive"}
              className="text-xs"
            >
              {stock.change >= 0 ? (
                <TrendingUp className="h-3 w-3 mr-1" />
              ) : (
                <TrendingDown className="h-3 w-3 mr-1" />
              )}
              {stock.change >= 0 ? "+" : ""}
              {stock.changePercent.toFixed(2)}%
            </Badge>
          </div>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {message && (
          <Alert variant={message.type === "error" ? "destructive" : "default"}>
            {message.type === "success" ? (
              <CheckCircle className="h-4 w-4" />
            ) : (
              <AlertCircle className="h-4 w-4" />
            )}
            <AlertDescription>{message.text}</AlertDescription>
          </Alert>
        )}

        <div>
          <Label htmlFor="quantity">Quantity (Shares)</Label>
          <Input
            id="quantity"
            type="number"
            value={quantity}
            onChange={(e) => setQuantity(e.target.value)}
            placeholder="Enter number of shares"
            min="1"
            disabled={isTrading}
          />
        </div>

        {quantity && (
          <div className="space-y-2 p-3 bg-gray-50 rounded-lg">
            <div className="flex justify-between text-sm">
              <span>Price per share:</span>
              <span>₹{stock.price.toFixed(2)}</span>
            </div>
            <div className="flex justify-between text-sm">
              <span>Buy Cost (with brokerage):</span>
              <span className="text-red-600">
                ₹{totalCost.toLocaleString("en-IN")}
              </span>
            </div>
            <div className="flex justify-between text-sm">
              <span>Sell Proceeds (after brokerage):</span>
              <span className="text-green-600">
                ₹{netProceeds.toLocaleString("en-IN")}
              </span>
            </div>
            {user && (
              <div className="flex justify-between text-sm border-t pt-2">
                <span>Available Balance:</span>
                <span className="font-medium">
                  ₹{user.balance.toLocaleString("en-IN")}
                </span>
              </div>
            )}
          </div>
        )}

        <div className="flex space-x-3">
          <Button
            onClick={handleBuy}
            disabled={isTrading || !quantity || parseInt(quantity) <= 0}
            className="flex-1 bg-green-600 hover:bg-green-700"
          >
            {isTrading ? (
              <Loader2 className="h-4 w-4 mr-2 animate-spin" />
            ) : (
              <ShoppingCart className="h-4 w-4 mr-2" />
            )}
            Buy
          </Button>

          <Button
            onClick={handleSell}
            disabled={isTrading || !quantity || parseInt(quantity) <= 0}
            variant="outline"
            className="flex-1"
          >
            {isTrading ? (
              <Loader2 className="h-4 w-4 mr-2 animate-spin" />
            ) : (
              <DollarSign className="h-4 w-4 mr-2" />
            )}
            Sell
          </Button>
        </div>

        <div className="text-xs text-gray-500 space-y-1">
          <p>• Brokerage: 0.5% on buy/sell transactions</p>
          <p>• Real-time pricing with simulated market data</p>
          <p>• Instant order execution</p>
        </div>
      </CardContent>
    </Card>
  );
}
