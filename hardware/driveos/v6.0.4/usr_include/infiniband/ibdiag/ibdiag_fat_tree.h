/*
 * Copyright (c) 2004-2021 Mellanox Technologies LTD. All rights reserved
 *
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
 * BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
 * ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 */


#ifndef IBDIAG_FT_H
#define IBDIAG_FT_H

#include <map>
#include <list>
#include <vector>
#include <sstream>
#include <bitset>

#include "ibdiag_types.h"
#include "ibdiag_fabric_errs.h"
#include "infiniband/ibdm/Fabric.h"
#include "ibdiagnet/ibdiagapp_types.h"

class FTClassification;
typedef vector<FTClassification*> classifications_vec;

class FTClassification {
public:
    typedef set <const IBNode*> nodes_set;
    typedef vector<nodes_set> nodes_by_rank_vec;
    typedef list<const IBNode*> nodes_list;
private:
    typedef map<int, nodes_list> distance_to_nodes_map;
    typedef map<const IBNode*, int> nodes_by_distance_map;
    const IBNode* GetLeafToClassify(const classifications_vec &classifications,
                                      const nodes_list &nodes);
    int GetMaxTresholdDistance() const;
    void ClassifyByDistance(const IBNode &leaf);
    int SetNodesRanks();
    int Set2L_FTRanks();
    int Set3L_FTRanks();
    int Set4L_FTRanks();
    void SetRankToNodes(const nodes_list &inList,   nodes_set &outSet) const;
    int Set4L_DistanceToRanks(int distance, int compareDistance);
    bool EqualRanks(const nodes_set &one, const nodes_set &other);
    bool EqualsTo(const FTClassification &other) const;
    int CalculateTreshold() const;
    string ToString () const;

    struct SearchData {
       const IBNode* p_node;
       int distance;

       SearchData(const IBNode* p_inNode = NULL, int inDist = 0):
            p_node(p_inNode), distance(inDist) {}
    };

    int maxDistance;
    int maxTresholdDistance;
    distance_to_nodes_map distanceToNodesMap;
    nodes_by_distance_map nodesByDistanceMap;
    nodes_by_rank_vec   nodesByRank;
    stringstream lastError;

public:
    FTClassification();

    int Classify(const IBNode &fromLeaf);
    int CheckDistanceTo(const IBNode &node, bool &inDistance) const;
    const IBNode* GetLeafToClassify(const classifications_vec &classifications);
    int CountEquals(const classifications_vec &classifications) const;
    void SwapRanks(nodes_by_rank_vec &ranks);
    string GetLastError() const { return lastError.str(); }
      //leaves
};

class FTClassificationHandler {
private:
    classifications_vec classifications;

public:
    ~FTClassificationHandler();

    FTClassification* GetNewClassification();
    const classifications_vec& GetClassifications() const { return classifications; }
};

typedef std::pair<int, int> LinksData;
class FTTopology;
class FTNeighborhood {
public:
    typedef std::map<int, std::vector<uint64_t> >  link_to_nodes_map;

public:
    FTNeighborhood(FTTopology &inT, size_t inId = -1, size_t inRank = -1):
                topology(inT), id(inId), rank(inRank){}
    ~FTNeighborhood() {}
    string GetLastError() const { return lastErrorStream.str(); }
    size_t GetId() const { return id; }
    void AddToUpNodes(const FTClassification::nodes_list &nodes) { AddNodes(nodes, true); }
    void AddToDownNodes(const FTClassification::nodes_list &nodes){ AddNodes(nodes, false); }
    int CheckUpDownLinks(list_p_fabric_general_err &errors, ostream *p_stream);
    int DumpToStream(ostream &stream) const;
    bool Contains(const IBNode *p_node) const {
            return (up.find(p_node) != up.end() ||
                    down.find(p_node) != down.end());
    }

private:
    int DumpNodesToStream(ostream &stream,
                        const FTClassification::nodes_set &nodes, const char *p_name) const;
    int CheckSetLinks(const FTClassification::nodes_set &nodes, size_t nodesRank, bool uplinks,
                    list_p_fabric_general_err &errors, ostream *p_stream);
    void SetLinksReport(ostream *p_stream, const link_to_nodes_map &linksToNodesMap,
                        size_t nodesRank, bool uplinks) const;

    //may be used in a future
    int CalculateUpDownLinks(LinksData &linksData);
    int CalculateUpDownLinks(const FTClassification::nodes_set &nodes, int &linksCount, bool isUp);

    void AddNodes(const FTClassification::nodes_list &nodes, bool isUp);
    bool IsWarning(size_t nodesRank, bool uplinks) const;
    void ReportToStream(ostream &stream, const link_to_nodes_map &map,
                        size_t maxInLine, const string &linkType) const;

    //todo consider use lists
    FTClassification::nodes_set up;
    FTClassification::nodes_set down;

    FTTopology &topology;
    size_t id;
    size_t rank;
    stringstream lastErrorStream;
};

struct FTLinkIssue {
   enum IssueType {Missing, Invalid, Unknown};

   const IBNode *p_node1;
   phys_port_t port1;
   size_t rank1;
   const IBNode *p_node2;
   phys_port_t port2;
   size_t rank2;
   IssueType type;

   FTLinkIssue(const IBNode *inNode1, phys_port_t inPort1, size_t inRank1,
            const IBNode*inNode2, phys_port_t inPort2, size_t inRank2,
            IssueType inType): p_node1(inNode1), port1(inPort1), rank1(inRank1),
                            p_node2(inNode2), port2(inPort2), rank2(inRank2),
                            type(inType) {}

   FTLinkIssue(const IBNode *inNode1, phys_port_t inPort1,
            const IBNode*inNode2, phys_port_t inPort2,
            size_t rank): p_node1(inNode1), port1(inPort1), rank1(rank),
                        p_node2(inNode2), port2(inPort2), rank2(rank),
                        type(Invalid) {}

   FTLinkIssue(const IBNode *inNode1, const IBNode*inNode2):
            p_node1(inNode1), port1(0), rank1(-1),
            p_node2(inNode2), port2(0), rank2(-1), type(Missing) {}

   FTLinkIssue():p_node1(NULL), port1(0), rank1(-1),
                p_node2(NULL), port2(0), rank2(-1), type(Unknown) {}
};

#define FT_UP_HOP_SET_SIZE 2048
struct FTUpHopSet {
    typedef std::bitset<FT_UP_HOP_SET_SIZE> bit_set;

    FTUpHopSet():encountered(0) { upNodesBitSet.reset(); }
    bool IsSubset(const FTUpHopSet &other) const {
        return ((this->upNodesBitSet | other.upNodesBitSet) == other.upNodesBitSet);
    }
    bit_set Complement(const FTUpHopSet &other) const {
        return (~other.upNodesBitSet & this->upNodesBitSet);
    }
    void AddDownNodes(const FTUpHopSet &other);

    int encountered;
    bit_set upNodesBitSet;
    FTClassification::nodes_list downNodes;
};

typedef std::vector <FTNeighborhood*> neighborhoods_vec;
class FTUpHopHistogram {
private:
      typedef std::map<std::string, FTUpHopSet> up_hop_sets_map;
      typedef std::map<size_t, const IBNode*> index_to_node_map;
      typedef std::map<const IBNode*, size_t> node_to_index_map;
      typedef std::vector <FTLinkIssue> link_issue_vec;

public:
    FTUpHopHistogram(FTTopology &inT, size_t inR): topology(inT), rank(inR),
                                            bitSetMaxSize(0), encounteredTresHold(0) {}
    int Init();
    int CreateNeighborhoods(list_p_fabric_general_err &errors);
    string GetLastError() const { return lastErrorStream.str(); }

private:
    std::string GetHashCode(const FTUpHopSet::bit_set &bitSet) const;
    void InitNodeToIndexConverters(const FTClassification::nodes_set &nodes);
    int NodeToIndex(size_t &index, const IBNode &p_node);
    const IBNode *IndexToNode(size_t index);
    int GetEncounterdTreshold();
    int TryMergeSets(const FTUpHopSet &currentSet, FTUpHopSet &other, bool &isMerged);
    int TryMergeSet(const FTUpHopSet &currentSet, bool &isMerged);
    int TrySplitSet(const FTUpHopSet &currentSet, bool &isSplitted);
    int TrySplitSets(const FTUpHopSet &currentSet, FTUpHopSet &other, bool &isSplitted);
    bool IsMostEncountered(const FTUpHopSet &upHopSet)
                    { return (upHopSet.encountered >= this->GetEncounterdTreshold()); }
    int SetsToNeigborhoods(list_p_fabric_general_err &errors);
    int BitSetToNodes(const FTUpHopSet::bit_set &bitSet, FTClassification::nodes_list &nodes);
    int LinkIssuesReport(list_p_fabric_general_err &errors,
                        const neighborhoods_vec &neighborhoods);
    const FTNeighborhood *FindNeighborhood(const neighborhoods_vec &neighborhoods,
                                        const IBNode *p_node);
    int AddMissingLinkIssues(const FTUpHopSet::bit_set &upSet,
                        const FTClassification::nodes_list &downNodes);
    void AddIllegalLinkIssue(const FTLinkIssue &issue);
    int AddIllegalLinkIssues(size_t index, const FTClassification::nodes_list &downNodes);


    std::string UpHopSetToString(const FTUpHopSet &upHopSet) const;
    void CheckRootSwitchConnections(const IBNode &node);

    index_to_node_map indexToNodeMap;
    node_to_index_map nodeToIndexMap;

    stringstream lastErrorStream;
    up_hop_sets_map upHopSetsMap;
    FTTopology &topology;
    const size_t rank;
    size_t bitSetMaxSize;
    int encounteredTresHold;
    link_issue_vec linkIssueVec;
};

class FTTopology {
public:
    typedef std::map<LinksData,
                    FTClassification::nodes_list > links_to_nodes_map;
public:
    FTTopology (IBFabric& ref, ostream *pr = NULL):
                            fabric(ref), minimalRatio(0), p_stream(pr){}
    ~FTTopology();

    int Build(list_p_fabric_general_err &errors, string &lastError, int retries, int equalResults);
    int Validate(list_p_fabric_general_err &errors, string &lastError);
    int Dump() const;
    size_t GetLevels() const { return nodesByRank.size(); }
    std::string LevelsReport() const;
    bool IsReportedLinkIssue(const IBNode* p_node1, const IBNode* p_node2) const;
    void AddNewLinkIssue(const IBNode* p_node1, const IBNode* p_node2);
    double GetMimimalRatio() const { return minimalRatio; };
    LinksData GetSwitchLinksData(size_t rank, const IBNode &node);
    int SetNeighborhoodsOnRank(neighborhoods_vec &neighborhoods, size_t rank);
    const FTClassification::nodes_set *GetNodesOnRank(size_t rank);
    size_t GetNodeRank(const IBNode* p_node) const;
    bool IsLastRankNeighborhood(size_t rank) const;

    IBFabric& fabric;

private:
    typedef std::map<const IBNode*, LinksData> node_to_links_map;

    const IBNode* GetFirstLeaf() const;
    int CreateNeighborhoods(list_p_fabric_general_err &errors);
    LinksData CalculateSwitchUpDownLinks(size_t rank, const IBNode& node);
    int CheckNeighborhoodsUpDownLinks(list_p_fabric_general_err &errors);
    int DumpNodesToStream(ostream &stream) const;
    int DumpNeighborhoodsToStream(ostream &stream) const;
    int CreateNeighborhoodsOnRank(list_p_fabric_general_err &errors, size_t rank);
    int CheckUpDownLinks(list_p_fabric_general_err &errors);
    int CalculateUpDownLinksMinRatio();

    std::vector< std::vector <FTNeighborhood*> > neighborhoodsByRank;
    node_to_links_map nodeToLinksMap;
    FTClassification::nodes_by_rank_vec nodesByRank;
    PairsContainer<const IBNode*> reportedLinksIssues;
    double minimalRatio;
    ostream *p_stream;
    stringstream lastErrorStream;
};

class FTInvalidLinkError : public FabricErrGeneral {
public:
    FTInvalidLinkError(size_t inId1, size_t inId2,
                    const FTLinkIssue &inIssue, bool bIn);
    ~FTInvalidLinkError() {}

    string GetCSVErrorLine() { return ""; }
    string GetErrorLine();

private:
    int PortToInt(phys_port_t port) const {
        return static_cast<int>(port);
    }

    size_t id_1;
    size_t id_2;
    FTLinkIssue issue;
    bool isNeighborhood;
};

class FTMissingLinkError : public FabricErrGeneral {
public:
    FTMissingLinkError(size_t inId, const FTLinkIssue &inIssue, bool bIn):
                id(inId), issue(inIssue), isNeighborhood(bIn) {}
    ~FTMissingLinkError() {}

    string GetCSVErrorLine() { return ""; }
    string GetErrorLine(); 

private:
    size_t id;
    FTLinkIssue issue;
    bool isNeighborhood;
};

class FTConsultDumpFileError : public FabricErrGeneral {
public:
    FTConsultDumpFileError(){}
    ~FTConsultDumpFileError(){}

    string GetCSVErrorLine() { return ""; }
    string GetErrorLine() {
        return string("For more errors see the dump file: ") + FAT_TREE_DUMP_FILE;
    }
};
#endif          /* IBDIAG_FT_H */
