//+------------------------------------------------------------------+
//|                                                ForexAI_EA_v1.mq5 |
//|                                      ForexAI-EA Basic Framework |
//|                                       Claude AI Developer v1.0.0 |
//+------------------------------------------------------------------+
#property copyright "ForexAI-EA Project"
#property version   "1.00"
#property description "AI-Powered Expert Advisor - Basic Framework"

// Include required libraries
#include <Trade\Trade.mqh>

//--- Input parameters
input group "=== AI Server Settings ==="
input string AIServerHost = "localhost";           // AI Server Host
input int    AIServerPort = 8888;                  // AI Server Port
input int    ConnectionTimeout = 30;               // Connection timeout (seconds)

input group "=== Trading Settings ==="
input double RiskPercent = 1.5;                    // Risk per trade (%)
input int    MaxPositions = 4;                     // Maximum simultaneous positions
input int    MagicNumber = 123456;                 // Magic number for EA trades

input group "=== AI Settings ==="
input int    AnalysisPeriod = 50;                  // Bars for AI analysis
input double ConfidenceThreshold = 0.7;            // Minimum confidence for trades
input int    PredictionInterval = 300;             // Prediction interval (seconds)

//--- Global variables
CTrade trade;
datetime lastPredictionTime = 0;
bool isConnectedToAI = false;
double priceBuffer[];

//+------------------------------------------------------------------+
//| Simple Socket Client Class (Basic Implementation)                |
//+------------------------------------------------------------------+
class CSimpleSocket
{
private:
    int m_socket;
    bool m_connected;

public:
    CSimpleSocket() : m_socket(-1), m_connected(false) {}
    
    ~CSimpleSocket()
    {
        Close();
    }
    
    bool Connect(string host, int port, int timeout_ms = 30000)
    {
        // For basic implementation, we'll use a simplified approach
        // In production, this would use WinAPI socket functions
        Print("Attempting to connect to ", host, ":", port);
        
        // Simulate connection for now - will be replaced with actual socket code
        Sleep(100);
        m_connected = true;
        Print("✓ Connected to AI server (simulated)");
        return true;
    }
    
    bool Send(string data)
    {
        if (!m_connected) return false;
        
        Print("Sending data: ", StringSubstr(data, 0, 100), "...");
        Sleep(50); // Simulate network delay
        return true;
    }
    
    bool Recv(string &response, int timeout_ms = 30000)
    {
        if (!m_connected) return false;
        
        // Mock response for testing - will be replaced with actual socket receive
        response = "{\"status\":\"success\",\"message\":\"pong\",\"timestamp\":\"" + 
                  TimeToString(TimeCurrent()) + "\"}";
        
        Sleep(50); // Simulate network delay
        return true;
    }
    
    void Close()
    {
        if (m_connected)
        {
            Print("Closing AI server connection");
            m_connected = false;
        }
    }
    
    bool IsConnected() { return m_connected; }
};

//--- Global socket instance
CSimpleSocket aiSocket;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
    Print("=== ForexAI-EA Initialization ===");
    
    // Set magic number for trade object
    trade.SetExpertMagicNumber(MagicNumber);
    
    // Set array as timeseries
    ArraySetAsSeries(priceBuffer, true);
    ArrayResize(priceBuffer, AnalysisPeriod);
    
    // Test AI server connection
    if(TestAIConnection())
    {
        Print("✓ AI Server connection successful");
        isConnectedToAI = true;
    }
    else
    {
        Print("✗ AI Server connection failed - EA will run in manual mode");
        isConnectedToAI = false;
    }
    
    Print("EA initialization completed");
    return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
    Print("=== ForexAI-EA Shutdown ===");
    
    // Close AI connection
    if(isConnectedToAI)
    {
        aiSocket.Close();
        Print("AI Server connection closed");
    }
    
    Print("EA shutdown completed. Reason: ", reason);
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{
    // Update price buffer with latest data
    UpdatePriceBuffer();
    
    // Check if it's time for new AI prediction
    if(TimeCurrent() - lastPredictionTime < PredictionInterval)
        return;
    
    // Get AI prediction if connected
    if(isConnectedToAI)
    {
        int aiSignal = GetAIPrediction();
        
        if(aiSignal != 0)
        {
            ProcessAISignal(aiSignal);
            lastPredictionTime = TimeCurrent();
        }
    }
    else
    {
        // Try to reconnect to AI server periodically
        if(TimeCurrent() - lastPredictionTime > 60) // Try every minute
        {
            if(TestAIConnection())
            {
                isConnectedToAI = true;
                Print("✓ Reconnected to AI Server");
            }
            lastPredictionTime = TimeCurrent();
        }
    }
}

//+------------------------------------------------------------------+
//| Update price buffer with latest price data                       |
//+------------------------------------------------------------------+
void UpdatePriceBuffer()
{
    double closePrice[];
    ArraySetAsSeries(closePrice, true);
    
    int copied = CopyClose(_Symbol, PERIOD_CURRENT, 0, AnalysisPeriod, closePrice);
    if(copied == AnalysisPeriod)
    {
        ArrayCopy(priceBuffer, closePrice);
    }
}

//+------------------------------------------------------------------+
//| Test connection to AI server                                     |
//+------------------------------------------------------------------+
bool TestAIConnection()
{
    if(!aiSocket.Connect(AIServerHost, AIServerPort, ConnectionTimeout * 1000))
    {
        Print("Failed to connect to AI server: ", AIServerHost, ":", AIServerPort);
        return false;
    }
    
    // Send ping request
    string pingRequest = "{\"type\":\"ping\",\"timestamp\":\"" + 
                        TimeToString(TimeCurrent()) + "\"}";
    
    if(!aiSocket.Send(pingRequest))
    {
        Print("Failed to send ping request");
        aiSocket.Close();
        return false;
    }
    
    // Wait for response
    string response;
    if(!aiSocket.Recv(response, ConnectionTimeout * 1000))
    {
        Print("Failed to receive ping response");
        aiSocket.Close();
        return false;
    }
    
    // Parse response (simple check for now)
    if(StringFind(response, "pong") >= 0)
    {
        return true;
    }
    
    Print("Invalid ping response: ", response);
    aiSocket.Close();
    return false;
}

//+------------------------------------------------------------------+
//| Get AI prediction from server                                    |
//+------------------------------------------------------------------+
int GetAIPrediction()
{
    if(!isConnectedToAI)
        return 0;
    
    // Prepare prediction request
    string request = CreatePredictionRequest();
    
    // For now, create new connection for each request
    CSimpleSocket predictionSocket;
    if(!predictionSocket.Connect(AIServerHost, AIServerPort, ConnectionTimeout * 1000))
    {
        Print("Failed to connect for prediction");
        isConnectedToAI = false;
        return 0;
    }
    
    // Send prediction request
    if(!predictionSocket.Send(request))
    {
        Print("Failed to send prediction request");
        predictionSocket.Close();
        return 0;
    }
    
    // Receive response
    string response;
    if(!predictionSocket.Recv(response, ConnectionTimeout * 1000))
    {
        Print("Failed to receive prediction response");
        predictionSocket.Close();
        return 0;
    }
    
    predictionSocket.Close();
    
    // Parse and return prediction
    return ParsePredictionResponse(response);
}

//+------------------------------------------------------------------+
//| Create prediction request JSON                                   |
//+------------------------------------------------------------------+
string CreatePredictionRequest()
{
    string priceArray = "[";
    
    // Add price data to array
    for(int i = 0; i < AnalysisPeriod && i < ArraySize(priceBuffer); i++)
    {
        if(i > 0) priceArray += ",";
        priceArray += DoubleToString(priceBuffer[i], _Digits);
    }
    priceArray += "]";
    
    // Create JSON request
    string request = "{";
    request += "\"type\":\"prediction\",";
    request += "\"symbol\":\"" + _Symbol + "\",";
    request += "\"timeframe\":\"" + EnumToString(PERIOD_CURRENT) + "\",";
    request += "\"prices\":" + priceArray + ",";
    request += "\"timestamp\":\"" + TimeToString(TimeCurrent()) + "\"";
    request += "}";
    
    return request;
}

//+------------------------------------------------------------------+
//| Parse prediction response from AI server                         |
//+------------------------------------------------------------------+
int ParsePredictionResponse(string response)
{
    // For initial testing, return mock prediction
    // This will be replaced with actual JSON parsing
    
    Print("Received AI response: ", StringSubstr(response, 0, 200));
    
    // Mock prediction logic for testing
    static int mockCounter = 0;
    mockCounter++;
    
    // Simulate different predictions
    int prediction;
    double confidence;
    
    if(mockCounter % 3 == 0)
    {
        prediction = 1;  // Buy
        confidence = 0.75;
    }
    else if(mockCounter % 3 == 1)
    {
        prediction = -1; // Sell
        confidence = 0.80;
    }
    else
    {
        prediction = 0;  // Hold
        confidence = 0.65;
    }
    
    // Check confidence threshold
    if(confidence < ConfidenceThreshold)
    {
        Print("Low confidence prediction: ", confidence, " < ", ConfidenceThreshold);
        return 0;
    }
    
    Print("AI Prediction: ", prediction, " (Confidence: ", confidence, ")");
    return prediction;
}

//+------------------------------------------------------------------+
//| Process AI signal and execute trades                             |
//+------------------------------------------------------------------+
void ProcessAISignal(int signal)
{
    // Check if we can trade
    if(!IsTradeAllowed())
        return;
    
    // Close existing positions first (for this EA only)
    CloseExistingPositions();
    
    // Execute trade based on signal
    if(signal == 1) // Buy signal
    {
        OpenBuyPosition();
    }
    else if(signal == -1) // Sell signal
    {
        OpenSellPosition();
    }
    // signal == 0 means hold (do nothing)
}

//+------------------------------------------------------------------+
//| Check if trading is allowed                                      |
//+------------------------------------------------------------------+
bool IsTradeAllowed()
{
    // Check if trading is allowed
    if(!TerminalInfoInteger(TERMINAL_TRADE_ALLOWED))
    {
        Print("Trading is not allowed in terminal");
        return false;
    }
    
    // Check if EA trading is allowed
    if(!MQLInfoInteger(MQL_TRADE_ALLOWED))
    {
        Print("EA trading is not allowed");
        return false;
    }
    
    // Check if symbol is tradeable
    if(!SymbolInfoInteger(_Symbol, SYMBOL_TRADE_MODE))
    {
        Print("Trading is not allowed for ", _Symbol);
        return false;
    }
    
    // Check maximum positions limit
    int currentPositions = CountEAPositions();
    if(currentPositions >= MaxPositions)
    {
        Print("Maximum positions reached: ", currentPositions);
        return false;
    }
    
    return true;
}

//+------------------------------------------------------------------+
//| Count positions opened by this EA                                |
//+------------------------------------------------------------------+
int CountEAPositions()
{
    int count = 0;
    for(int i = 0; i < PositionsTotal(); i++)
    {
        if(PositionGetSymbol(i) == _Symbol && 
           PositionGetInteger(POSITION_MAGIC) == MagicNumber)
        {
            count++;
        }
    }
    return count;
}

//+------------------------------------------------------------------+
//| Close existing positions for this EA                             |
//+------------------------------------------------------------------+
void CloseExistingPositions()
{
    for(int i = PositionsTotal() - 1; i >= 0; i--)
    {
        if(PositionGetSymbol(i) == _Symbol && 
           PositionGetInteger(POSITION_MAGIC) == MagicNumber)
        {
            ulong ticket = PositionGetInteger(POSITION_TICKET);
            if(!trade.PositionClose(ticket))
            {
                Print("Failed to close position #", ticket, " Error: ", GetLastError());
            }
            else
            {
                Print("Closed position #", ticket);
            }
        }
    }
}

//+------------------------------------------------------------------+
//| Open buy position                                                |
//+------------------------------------------------------------------+
void OpenBuyPosition()
{
    double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
    
    // Calculate ATR for dynamic SL/TP
    double atr[];
    ArraySetAsSeries(atr, true);
    if(CopyBuffer(iATR(_Symbol, PERIOD_CURRENT, 14), 0, 0, 1, atr) != 1)
    {
        Print("Failed to get ATR value");
        return;
    }
    
    // Calculate position size based on risk
    double stopLoss = ask - (atr[0] * 2.0);
    double takeProfit = ask + (atr[0] * 3.0);
    double lotSize = CalculatePositionSize(ask, stopLoss);
    
    // Normalize values
    stopLoss = NormalizeDouble(stopLoss, _Digits);
    takeProfit = NormalizeDouble(takeProfit, _Digits);
    
    if(trade.Buy(lotSize, _Symbol, ask, stopLoss, takeProfit, "AI Buy Signal"))
    {
        Print("✓ Buy order placed: Lot=", lotSize, " SL=", stopLoss, " TP=", takeProfit);
    }
    else
    {
        Print("✗ Failed to place buy order. Error: ", GetLastError());
    }
}

//+------------------------------------------------------------------+
//| Open sell position                                               |
//+------------------------------------------------------------------+
void OpenSellPosition()
{
    double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
    
    // Calculate ATR for dynamic SL/TP
    double atr[];
    ArraySetAsSeries(atr, true);
    if(CopyBuffer(iATR(_Symbol, PERIOD_CURRENT, 14), 0, 0, 1, atr) != 1)
    {
        Print("Failed to get ATR value");
        return;
    }
    
    // Calculate position size based on risk
    double stopLoss = bid + (atr[0] * 2.0);
    double takeProfit = bid - (atr[0] * 3.0);
    double lotSize = CalculatePositionSize(bid, stopLoss);
    
    // Normalize values
    stopLoss = NormalizeDouble(stopLoss, _Digits);
    takeProfit = NormalizeDouble(takeProfit, _Digits);
    
    if(trade.Sell(lotSize, _Symbol, bid, stopLoss, takeProfit, "AI Sell Signal"))
    {
        Print("✓ Sell order placed: Lot=", lotSize, " SL=", stopLoss, " TP=", takeProfit);
    }
    else
    {
        Print("✗ Failed to place sell order. Error: ", GetLastError());
    }
}

//+------------------------------------------------------------------+
//| Calculate position size based on risk percentage                 |
//+------------------------------------------------------------------+
double CalculatePositionSize(double entryPrice, double stopLoss)
{
    double accountBalance = AccountInfoDouble(ACCOUNT_BALANCE);
    double riskAmount = accountBalance * RiskPercent / 100.0;
    
    double stopLossDistance = MathAbs(entryPrice - stopLoss);
    double tickValue = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
    double tickSize = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
    
    double lotSize = riskAmount / (stopLossDistance / tickSize * tickValue);
    
    // Normalize lot size
    double minLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
    double maxLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
    double lotStep = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
    
    lotSize = MathMax(minLot, MathMin(maxLot, 
              MathRound(lotSize / lotStep) * lotStep));
    
    return lotSize;
}

//+------------------------------------------------------------------+
//| Handle trading events                                            |
//+------------------------------------------------------------------+
void OnTradeTransaction(const MqlTradeTransaction& trans,
                       const MqlTradeRequest& request,
                       const MqlTradeResult& result)
{
    // Log trade transactions for this EA
    if(request.magic == MagicNumber)
    {
        Print("Trade Transaction: ", EnumToString(trans.type), 
              " Symbol: ", trans.symbol,
              " Volume: ", trans.volume,
              " Price: ", trans.price);
    }
}