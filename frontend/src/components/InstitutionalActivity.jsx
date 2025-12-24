import { useState, useEffect } from "react";
import axios from "axios";
import { motion } from "framer-motion";
import { Building2, TrendingUp, TrendingDown, RefreshCw, ArrowUpRight, ArrowDownRight } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { API_URL } from "@/config/api";

export default function InstitutionalActivity() {
  const [fiiDii, setFiiDii] = useState(null);
  const [bulkDeals, setBulkDeals] = useState([]);
  const [blockDeals, setBlockDeals] = useState([]);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    loadFromCache();
  }, []);

  const loadFromCache = () => {
    try {
      // Load from localStorage first
      const cached = localStorage.getItem('institutional_activity');

      if (cached) {
        const { data, timestamp } = JSON.parse(cached);
        const age = Date.now() - timestamp;

        // Use cached data if less than 30 minutes old (FII/DII updates slowly)
        if (age < 30 * 60 * 1000) {
          setFiiDii(data.fiiDii);
          setBulkDeals(data.bulkDeals || []);
          setBlockDeals(data.blockDeals || []);
          setIsLoading(false);
          return; // Don't fetch from API
        }
      }

      // No cache or expired - fetch fresh data
      fetchData();
    } catch (error) {
      console.error("Cache load error:", error);
      fetchData();
    }
  };

  const fetchData = async () => {
    setIsLoading(true);
    try {
      const [fiiRes, bulkRes, blockRes] = await Promise.all([
        axios.get(`${API_URL}/market/fii-dii`),
        axios.get(`${API_URL}/market/bulk-deals`),
        axios.get(`${API_URL}/market/block-deals`)
      ]);

      const fiiData = fiiRes.data;
      const bulkData = bulkRes.data.deals || bulkRes.data || [];
      const blockData = blockRes.data.deals || blockRes.data || [];

      setFiiDii(fiiData);
      setBulkDeals(bulkData);
      setBlockDeals(blockData);

      // Cache in localStorage
      localStorage.setItem('institutional_activity', JSON.stringify({
        data: {
          fiiDii: fiiData,
          bulkDeals: bulkData,
          blockDeals: blockData
        },
        timestamp: Date.now()
      }));
    } catch (error) {
      console.error("Error fetching institutional data:", error);
    } finally {
      setIsLoading(false);
    }
  };

  const formatCrores = (value) => {
    if (!value) return "N/A";
    const absValue = Math.abs(value);
    if (absValue >= 100) {
      return `₹${(value / 100).toFixed(2)}K Cr`;
    }
    return `₹${value.toFixed(2)} Cr`;
  };

  const FlowCard = ({ title, buy, sell, net, icon: Icon }) => {
    const isPositive = net >= 0;
    return (
      <div className="p-4 rounded-lg bg-surface-highlight">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2">
            <Icon className="w-4 h-4 text-text-secondary" />
            <span className="text-sm font-medium text-text-primary">{title}</span>
          </div>
          <Badge className={`${isPositive ? 'bg-success/10 text-success' : 'bg-danger/10 text-danger'} border-0`}>
            {isPositive ? 'Net Buyer' : 'Net Seller'}
          </Badge>
        </div>
        <div className="grid grid-cols-3 gap-3 text-center">
          <div>
            <p className="text-xs text-text-secondary">Buy</p>
            <p className="font-data text-success">{formatCrores(buy)}</p>
          </div>
          <div>
            <p className="text-xs text-text-secondary">Sell</p>
            <p className="font-data text-danger">{formatCrores(sell)}</p>
          </div>
          <div>
            <p className="text-xs text-text-secondary">Net</p>
            <p className={`font-data font-bold ${isPositive ? 'text-success' : 'text-danger'}`}>
              {isPositive ? '+' : ''}{formatCrores(net)}
            </p>
          </div>
        </div>
      </div>
    );
  };

  return (
    <Card className="card-surface">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="text-sm font-heading flex items-center gap-2">
            <Building2 className="w-4 h-4 text-ai-accent" />
            Institutional Activity
          </CardTitle>
          <Button variant="ghost" size="sm" onClick={fetchData} className="h-7 w-7 p-0">
            <RefreshCw className={`w-3 h-3 ${isLoading ? 'animate-spin' : ''}`} />
          </Button>
        </div>
      </CardHeader>
      <CardContent>
        {isLoading ? (
          <div className="flex items-center justify-center py-12">
            <RefreshCw className="w-6 h-6 animate-spin text-text-secondary" />
          </div>
        ) : (
          <Tabs defaultValue="flows" className="w-full">
            <TabsList className="grid w-full grid-cols-3 mb-4">
              <TabsTrigger value="flows">FII/DII</TabsTrigger>
              <TabsTrigger value="bulk">Bulk Deals</TabsTrigger>
              <TabsTrigger value="block">Block Deals</TabsTrigger>
            </TabsList>

            <TabsContent value="flows" className="space-y-3">
              {fiiDii ? (
                <>
                  <FlowCard
                    title="FII (Foreign)"
                    buy={fiiDii.fii?.buy_value || fiiDii.fii?.buy || fiiDii.fii_buy}
                    sell={fiiDii.fii?.sell_value || fiiDii.fii?.sell || fiiDii.fii_sell}
                    net={fiiDii.fii?.net_value || fiiDii.fii?.net || fiiDii.fii_net}
                    icon={Building2}
                  />
                  <FlowCard
                    title="DII (Domestic)"
                    buy={fiiDii.dii?.buy_value || fiiDii.dii?.buy || fiiDii.dii_buy}
                    sell={fiiDii.dii?.sell_value || fiiDii.dii?.sell || fiiDii.dii_sell}
                    net={fiiDii.dii?.net_value || fiiDii.dii?.net || fiiDii.dii_net}
                    icon={Building2}
                  />

                  {/* Net Flow Summary */}
                  <div className="p-4 rounded-lg bg-primary/10 border border-primary/20">
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-text-primary">Total Net Flow</span>
                      <div className="flex items-center gap-2">
                        {((fiiDii.fii?.net_value || fiiDii.fii?.net || 0) + (fiiDii.dii?.net_value || fiiDii.dii?.net || 0)) >= 0 ? (
                          <ArrowUpRight className="w-5 h-5 text-success" />
                        ) : (
                          <ArrowDownRight className="w-5 h-5 text-danger" />
                        )}
                        <span className={`font-data font-bold ${
                          ((fiiDii.fii?.net_value || fiiDii.fii?.net || 0) + (fiiDii.dii?.net_value || fiiDii.dii?.net || 0)) >= 0 ? 'text-success' : 'text-danger'
                        }`}>
                          {formatCrores((fiiDii.fii?.net_value || fiiDii.fii?.net || 0) + (fiiDii.dii?.net_value || fiiDii.dii?.net || 0))}
                        </span>
                      </div>
                    </div>
                    <p className="text-xs text-text-secondary mt-1">
                      Date: {fiiDii.date || 'Latest'}
                    </p>
                  </div>
                </>
              ) : (
                <p className="text-text-secondary text-center py-8">No FII/DII data available</p>
              )}
            </TabsContent>

            <TabsContent value="bulk" className="max-h-[300px] overflow-y-auto">
              {bulkDeals.length === 0 ? (
                <p className="text-text-secondary text-center py-8">No bulk deals today</p>
              ) : (
                <div className="space-y-2">
                  {bulkDeals.slice(0, 10).map((deal, idx) => (
                    <motion.div
                      key={idx}
                      initial={{ opacity: 0, x: -10 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: idx * 0.05 }}
                      className="p-3 rounded-lg bg-surface-highlight"
                    >
                      <div className="flex items-center justify-between">
                        <div>
                          <span className="font-data font-medium text-text-primary">{deal.symbol}</span>
                          <p className="text-xs text-text-secondary truncate max-w-[150px]">{deal.client}</p>
                        </div>
                        <div className="text-right">
                          <Badge className={`${deal.type === 'BUY' ? 'bg-success/10 text-success' : 'bg-danger/10 text-danger'} border-0`}>
                            {deal.type}
                          </Badge>
                          <p className="text-xs text-text-secondary mt-1">
                            {deal.quantity?.toLocaleString()} @ ₹{deal.price}
                          </p>
                        </div>
                      </div>
                    </motion.div>
                  ))}
                </div>
              )}
            </TabsContent>

            <TabsContent value="block" className="max-h-[300px] overflow-y-auto">
              {blockDeals.length === 0 ? (
                <p className="text-text-secondary text-center py-8">No block deals today</p>
              ) : (
                <div className="space-y-2">
                  {blockDeals.slice(0, 10).map((deal, idx) => (
                    <motion.div
                      key={idx}
                      initial={{ opacity: 0, x: -10 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: idx * 0.05 }}
                      className="p-3 rounded-lg bg-surface-highlight"
                    >
                      <div className="flex items-center justify-between">
                        <div>
                          <span className="font-data font-medium text-text-primary">{deal.symbol}</span>
                          <p className="text-xs text-text-secondary">{deal.client || 'Institution'}</p>
                        </div>
                        <div className="text-right">
                          <p className="font-data text-text-primary">₹{deal.value?.toLocaleString() || 'N/A'} Cr</p>
                          <p className="text-xs text-text-secondary">
                            {deal.quantity?.toLocaleString()} shares
                          </p>
                        </div>
                      </div>
                    </motion.div>
                  ))}
                </div>
              )}
            </TabsContent>
          </Tabs>
        )}
      </CardContent>
    </Card>
  );
}
