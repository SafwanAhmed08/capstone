// DDoS TCP SYN Flood Simulation - Training Data Profile Match
#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/applications-module.h"

using namespace ns3;
NS_LOG_COMPONENT_DEFINE("DDoSProfile");

class DDoSApp : public Application {
private:
    Ptr<Socket> m_socket;
    Address m_peer;
    uint32_t m_sent;
    uint32_t m_synTarget;    
    uint32_t m_rstTarget;    
    EventId m_event;

    void SendSynPacket() {
        if (m_sent < m_synTarget) {
            Ptr<Socket> synSocket = Socket::CreateSocket(GetNode(), TcpSocketFactory::GetTypeId());
            synSocket->Connect(m_peer);
            Simulator::Schedule(MicroSeconds(1), &Socket::Close, synSocket);
            m_sent++;
            // Send at 2000 packets/sec = 0.5ms interval
            Simulator::Schedule(MicroSeconds(500), &DDoSApp::SendSynPacket, this);
        } else if (m_sent < m_synTarget + m_rstTarget) {
            SendRstPacket();
        }
    }

    void SendRstPacket() {
        if (m_sent < m_synTarget + m_rstTarget) {
            Ptr<Socket> rstSocket = Socket::CreateSocket(GetNode(), TcpSocketFactory::GetTypeId());
            rstSocket->Connect(m_peer);
            rstSocket->Close(); // Immediate close generates RST
            m_sent++;
            // Continue at same rate
            Simulator::Schedule(MicroSeconds(500), &DDoSApp::SendRstPacket, this);
        }
    }

public:
    // Each attacker sends 20 packets over 1 second (100 attackers * 20 = 2000 packets = 2000/sec)
    DDoSApp() : m_socket(0), m_sent(0), m_synTarget(12), m_rstTarget(8) {} // 60% SYN, 40% RST
    
    void Setup(Address addr) { m_peer = addr; }
    
    virtual void StartApplication() {
        m_sent = 0;
        SendSynPacket();
    }
    
    virtual void StopApplication() {
        if (m_event.IsRunning()) Simulator::Cancel(m_event);
        if (m_socket) m_socket->Close();
    }
};

int main(int argc, char *argv[]) {
    // Training Profile: 1192 attackers → 810 targets (simplified to 100→3)
    NodeContainer attackers, targets;
    attackers.Create(100);  // Represents 1192 distributed sources
    targets.Create(3);      // Represents focused target set

    InternetStackHelper internet;
    internet.Install(attackers);
    internet.Install(targets);

    // High-speed links for burst traffic (2000 packets/sec)
    PointToPointHelper p2p;
    p2p.SetDeviceAttribute("DataRate", StringValue("100Mbps"));
    p2p.SetChannelAttribute("Delay", StringValue("1ms"));

    // Connect attackers to targets with minimal routing
    Ipv4AddressHelper address;
    std::vector<Ipv4InterfaceContainer> interfaces;
    
    for (uint32_t i = 0; i < attackers.GetN(); ++i) {
        uint32_t targetIdx = i % targets.GetN(); // Distribute load
        NodeContainer pair(attackers.Get(i), targets.Get(targetIdx));
        NetDeviceContainer devices = p2p.Install(pair);
        
        address.SetBase(Ipv4Address((10 << 24) | (1 << 16) | ((i+1) << 8)), "255.255.255.0");
        interfaces.push_back(address.Assign(devices));
        address.NewNetwork();
    }

    // TCP sinks on targets (ports 80, 443, 8080 for diversity)
    uint16_t ports[] = {80, 443, 8080};
    for (uint32_t i = 0; i < targets.GetN(); ++i) {
        PacketSinkHelper sink("ns3::TcpSocketFactory", 
                             InetSocketAddress(Ipv4Address::GetAny(), ports[i]));
        ApplicationContainer sinkApp = sink.Install(targets.Get(i));
        sinkApp.Start(Seconds(0.0));
        sinkApp.Stop(Seconds(3.0));
    }

    // DDoS attack: Each attacker sends 20 packets (2000 total over 1 second = 2000/sec)
    for (uint32_t i = 0; i < attackers.GetN(); ++i) {
        Ptr<DDoSApp> app = CreateObject<DDoSApp>();
        uint32_t targetIdx = i % targets.GetN();
        Ipv4Address targetIP = interfaces[i].GetAddress(1); // Target is second address in pair
        app->Setup(InetSocketAddress(targetIP, ports[targetIdx]));
        
        attackers.Get(i)->AddApplication(app);
        app->SetStartTime(Seconds(1.0 + i * 0.001)); // Start spread over 100ms
        app->SetStopTime(Seconds(3.0));
    }

    // Configure TCP for training profile match (71.5 avg bytes)
    Config::SetDefault("ns3::TcpSocket::SegmentSize", UintegerValue(40));
    Config::SetDefault("ns3::TcpSocket::SndBufSize", UintegerValue(4096));
    Config::SetDefault("ns3::TcpSocket::RcvBufSize", UintegerValue(4096));

    // Capture all traffic for analysis
    p2p.EnablePcapAll("ddos-profile");

    Ipv4GlobalRoutingHelper::PopulateRoutingTables();
    Simulator::Stop(Seconds(5.0)); // Run for 5 seconds total
    
    std::cout << "🚨 DDoS Profile Simulation Starting...\n";
    std::cout << "Target: 2000 packets/sec for 1 second, 60% SYN, 40% RST, 0% success\n";
    
    Simulator::Run();
    Simulator::Destroy();
    return 0;
}