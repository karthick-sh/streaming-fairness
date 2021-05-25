// Calomel.org Proxy Auto-Config
//
 
//
// Define the network paths (direct, proxy and deny)
//
 
// Default connection
var direct = "DIRECT";
 
// Alternate Proxy Server
var proxy = "PROXY 0.0.0.0:9001";
 
// Default localhost for denied connections
var deny = "PROXY 0.0.0.0:65535";
 
//
// Proxy Logic
//
 
function FindProxyForURL(url, host)
{
    if (url.substring(0, 3) === "ws:" || url.substring(0, 4) === "wss:")
    {
        return direct;
    } else {
        return proxy;
    }
 
    return direct;
}