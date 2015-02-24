/**
 *  @file   LCContent/src/LCContentFast/SoftClusterMergingAlgorithmFast.cc
 * 
 *  @brief  Implementation of the soft cluster merging algorithm class.
 * 
 *  $Log: $
 */

#include "Pandora/AlgorithmHeaders.h"

#include "LCHelpers/ClusterHelper.h"
#include "LCHelpers/SortingHelper.h"

#include "LCContentFast/KDTreeLinkerAlgoT.h"
#include "LCContentFast/SoftClusterMergingAlgorithmFast.h"
#include "LCContentFast/QuickUnion.h"

#include <memory>

using namespace pandora;

namespace lc_content_fast
{

SoftClusterMergingAlgorithm::SoftClusterMergingAlgorithm() :
    m_shouldUseCurrentClusterList(true),
    m_updateCurrentTrackClusterAssociations(true),
    m_maxHitsInSoftCluster(5),
    m_maxLayersSpannedBySoftCluster(3),
    m_maxHadEnergyForSoftClusterNoTrack(2.f),
    m_minClusterHadEnergy(0.25f),
    m_minClusterEMEnergy(0.025f),
    m_minCosOpeningAngle(0.f),
    m_minHitsInCluster(5),
    m_closestDistanceCut0(50.f),
    m_closestDistanceCut1(100.f),
    m_innerLayerCut1(20),
    m_closestDistanceCut2(250.f),
    m_innerLayerCut2(40),
    m_maxClusterDistanceFine(100.f),
    m_maxClusterDistanceCoarse(250.f)
{
}

//------------------------------------------------------------------------------------------------------------------------------------------

SoftClusterMergingAlgorithm::~SoftClusterMergingAlgorithm()
{

}

//------------------------------------------------------------------------------------------------------------------------------------------

StatusCode SoftClusterMergingAlgorithm::Run()
{
    // need to do some of the searching with vectors (hash table too slow)
    HitKDTreeByIndex hits_kdtree_byindex;
    std::vector<HitKDNodeByIndex> hit_nodes_by_index;
    std::vector<const CaloHit*> hits_by_index;
    std::vector<unsigned int> hit_index_to_cluster_index;

    // save local kd-trees of each cluster that we will update over time
    // this will hopefully speed up cluster distance calculations
	std::vector<std::unique_ptr<HitKDTree> > trees_by_cluster_index;

    // get the *starting* cluster list
    ClusterList clusterList;
    ClusterListToNameMap clusterListToNameMap;
    this->GetInputClusters(clusterList, clusterListToNameMap);	

    ClusterVector clusterVector(clusterList.begin(), clusterList.end());
    std::sort(clusterVector.begin(), clusterVector.end(), lc_content::SortingHelper::SortClustersByInnerLayer);
    QuickUnion quickUnion(clusterVector.size());
    trees_by_cluster_index.reserve(clusterVector.size());

	int index(-1);
    for(const Cluster *const pCluster : clusterVector) {
        ++index;
	    trees_by_cluster_index.push_back(std::unique_ptr<HitKDTree>(new HitKDTree()));
	    const auto& tree = trees_by_cluster_index.back();
        std::array<float,3> minpos{ {0.0f,0.0f,0.0f} }, maxpos{ {0.0f,0.0f,0.0f} };
        std::vector<HitKDNode> nodes_for_local_tree;
        CaloHitList temp;      
        pCluster->GetOrderedCaloHitList().GetCaloHitList(temp);
        unsigned nhits = 0;
        for( auto* hit : temp ) {
            const CartesianVector& pos = hit->GetPositionVector();
            nodes_for_local_tree.emplace_back(hit,pos.GetX(),pos.GetY(),pos.GetZ());
            if( nhits == 0 ) {
                minpos[0] = pos.GetX(); minpos[1] = pos.GetY(); minpos[2] = pos.GetZ();
                maxpos[0] = pos.GetX(); maxpos[1] = pos.GetY(); maxpos[2] = pos.GetZ();
            } else {
                minpos[0] = std::min((float)pos.GetX(),minpos[0]);
                minpos[1] = std::min((float)pos.GetY(),minpos[1]);
                minpos[2] = std::min((float)pos.GetZ(),minpos[2]);
                maxpos[0] = std::max((float)pos.GetX(),maxpos[0]);
                maxpos[1] = std::max((float)pos.GetY(),maxpos[1]);
                maxpos[2] = std::max((float)pos.GetZ(),maxpos[2]);
            }
            hits_by_index.emplace_back(hit);
            hit_index_to_cluster_index.emplace_back(index);	
            ++nhits;
        }
        KDTreeCube clusterBoundingBox(minpos[0],maxpos[0],
                                      minpos[1],maxpos[1],
                                      minpos[2],maxpos[2]);
        tree->build(nodes_for_local_tree,clusterBoundingBox);
    }
    KDTreeCube hitsByIndexBoundingRegion =
      fill_and_bound_3d_kd_tree_by_index(hits_by_index,hit_nodes_by_index);
    hits_kdtree_byindex.build(hit_nodes_by_index,hitsByIndexBoundingRegion);
    hit_nodes_by_index.clear();
	
    index = -1;
    for (const Cluster *const pDaughterCluster : clusterVector)
    {
        ++index;

        if (!this->IsSoftCluster(pDaughterCluster))
            continue;

        int bestParentIndex(-1);
        float bestParentClusterEnergy(0.);
        float minDistanceSquared(std::numeric_limits<float>::max());
		const float searchDistance((PandoraContentApi::GetGeometry(*this)->GetHitTypeGranularity(pDaughterCluster->GetOuterLayerHitType()) <= FINE) ?
          m_maxClusterDistanceFine : m_maxClusterDistanceCoarse);

        CaloHitList theseHits;
        pDaughterCluster->GetOrderedCaloHitList().GetCaloHitList(theseHits);

	    for (const CaloHit *const pCaloHitI : theseHits)
        {
			const CartesianVector &positionVectorI(pCaloHitI->GetPositionVector());
			
	        // find our nearby clusters
			KDTreeCube hitSearchRegion = build_3d_kd_search_region(pCaloHitI,
                                                                   searchDistance,
                                                                   searchDistance,
                                                                   searchDistance );
	        std::vector<HitKDNodeByIndex> found_hits;
	        hits_kdtree_byindex.search(hitSearchRegion,found_hits);

	        for( auto& found_hit : found_hits ) {
				//in this loop, run all neighbour indices through quickUnion to keep track of merged clusters
	            unsigned int parentIndex = quickUnion.Find(hit_index_to_cluster_index.at(found_hit.data));
				// make sure this is a neighbour
	            if( index == static_cast<int>(parentIndex) )  
	                continue;
	            
				const Cluster *const pParentCluster = clusterVector.at(parentIndex);
                const float clusterEnergy(pParentCluster->GetHadronicEnergy());
                
                if (clusterEnergy < m_minClusterHadEnergy)
                    continue;
                
                if (pParentCluster->GetNCaloHits() <= m_maxHitsInSoftCluster)
                    continue;
                
				// find the NN in the parent cluster and test
				const auto& hit_tree = trees_by_cluster_index.at(parentIndex);
	            float parent_distance = std::numeric_limits<float>::max();
	            HitKDNode daughter_point(pCaloHitI,positionVectorI.GetX(),positionVectorI.GetY(),positionVectorI.GetZ());
	            HitKDNode* theresult = nullptr;
	            hit_tree->findNearestNeighbour(daughter_point,theresult,parent_distance);	    
	            if( nullptr != theresult && parent_distance != std::numeric_limits<float>::max() ){
					const float distanceSquared(parent_distance*parent_distance);
                    if ((distanceSquared < minDistanceSquared) || ((distanceSquared == minDistanceSquared) && (clusterEnergy > bestParentClusterEnergy)))
                    {
                        minDistanceSquared = distanceSquared;
                        bestParentClusterEnergy = clusterEnergy;
                        bestParentIndex = parentIndex;
                    }
				}
	        }
		}

        if ((bestParentIndex >= 0) && this->CanMergeSoftCluster(pDaughterCluster, sqrt(minDistanceSquared)))
        {
            const Cluster *const pBestParentCluster = clusterVector.at(bestParentIndex);
			this->MergeClusters(pBestParentCluster, pDaughterCluster, clusterListToNameMap);
            quickUnion.Unite(index, bestParentIndex);
			// update the parent cluster kd tree
	        auto& tree = trees_by_cluster_index.at(bestParentIndex);
	        tree.reset(new HitKDTree());
	        CaloHitList temp;
	        std::vector<HitKDNode> hit_nodes;
	        pBestParentCluster->GetOrderedCaloHitList().GetCaloHitList(temp);
	        KDTreeCube clusterBoundingBox =
	          fill_and_bound_3d_kd_tree(temp,hit_nodes);
	        tree->build(hit_nodes,clusterBoundingBox);
        }
    }

    return STATUS_CODE_SUCCESS;
}

//------------------------------------------------------------------------------------------------------------------------------------------

void SoftClusterMergingAlgorithm::GetInputClusters(ClusterList &clusterList, ClusterListToNameMap &clusterListToNameMap) const
{
    if (m_shouldUseCurrentClusterList)
    {
        std::string clusterListName;
        const ClusterList *pClusterList = nullptr;
        PANDORA_THROW_RESULT_IF(STATUS_CODE_SUCCESS, !=, PandoraContentApi::GetCurrentList(*this, pClusterList, clusterListName));

        clusterList.insert(pClusterList->begin(), pClusterList->end());
        clusterListToNameMap.insert(ClusterListToNameMap::value_type(pClusterList, clusterListName));

        if (m_updateCurrentTrackClusterAssociations)
        {
            PANDORA_THROW_RESULT_IF(STATUS_CODE_SUCCESS, !=, PandoraContentApi::RunDaughterAlgorithm(*this, m_trackClusterAssociationAlgName));
        }
    }

    for (const std::string &listName : m_additionalClusterListNames)
    {
        const ClusterList *pClusterList = nullptr;

        if (STATUS_CODE_SUCCESS == PandoraContentApi::GetList(*this, listName, pClusterList))
        {
            clusterList.insert(pClusterList->begin(), pClusterList->end());
            clusterListToNameMap.insert(ClusterListToNameMap::value_type(pClusterList, listName));
        }
        else
        {
            std::cout << "SoftClusterMergingAlgorithm: Failed to obtain cluster list " << listName << std::endl;
        }
    }
}

//------------------------------------------------------------------------------------------------------------------------------------------

bool SoftClusterMergingAlgorithm::IsSoftCluster(const Cluster *const pDaughterCluster) const
{
    // Note the cuts applied here are order-dependent - use the order defined in original version of pandora
    const unsigned int nCaloHits(pDaughterCluster->GetNCaloHits());

    if (0 == nCaloHits)
        return false;

    // ATTN: Added this cut to prevent merging multiple track-seeded clusters
    if (!pDaughterCluster->GetAssociatedTrackList().empty())
        return false;

    bool isSoftCluster(false);

    if (nCaloHits <= m_maxHitsInSoftCluster)
    {
        isSoftCluster = true;
    }
    else if ((pDaughterCluster->GetOuterPseudoLayer() - pDaughterCluster->GetInnerPseudoLayer()) < m_maxLayersSpannedBySoftCluster)
    {
        isSoftCluster = true;
    }
    else if (pDaughterCluster->GetHadronicEnergy() < m_maxHadEnergyForSoftClusterNoTrack)
    {
        isSoftCluster = true;
    }

    if (pDaughterCluster->GetHadronicEnergy() < m_minClusterHadEnergy)
        isSoftCluster = true;

    if (pDaughterCluster->IsPhotonFast(this->GetPandora()) && (pDaughterCluster->GetElectromagneticEnergy() > m_minClusterEMEnergy))
        isSoftCluster = false;

    return isSoftCluster;
}

//------------------------------------------------------------------------------------------------------------------------------------------

bool SoftClusterMergingAlgorithm::CanMergeSoftCluster(const Cluster *const pDaughterCluster, const float closestDistance) const
{
    if (closestDistance < m_closestDistanceCut0)
        return true;

    const unsigned int daughterInnerLayer(pDaughterCluster->GetInnerPseudoLayer());

    if ((closestDistance < m_closestDistanceCut1) && (daughterInnerLayer > m_innerLayerCut1))
        return true;

    if ((closestDistance < m_closestDistanceCut2) && (daughterInnerLayer > m_innerLayerCut2))
        return true;

    const float distanceCut((PandoraContentApi::GetGeometry(*this)->GetHitTypeGranularity(pDaughterCluster->GetOuterLayerHitType()) <= FINE) ?
        m_maxClusterDistanceFine : m_maxClusterDistanceCoarse);

    if (closestDistance > distanceCut)
        return false;

    return ((pDaughterCluster->GetHadronicEnergy() < m_minClusterHadEnergy) || (pDaughterCluster->GetNCaloHits() < m_minHitsInCluster));
}

//------------------------------------------------------------------------------------------------------------------------------------------

void SoftClusterMergingAlgorithm::MergeClusters(const Cluster *const pParentCluster, const Cluster *const pDaughterCluster,
    const ClusterListToNameMap &clusterListToNameMap) const
{
    if (clusterListToNameMap.size() > 1)
    {
        std::string parentListName, daughterListName;
        PANDORA_THROW_RESULT_IF(STATUS_CODE_SUCCESS, !=, this->GetClusterListName(pParentCluster, clusterListToNameMap, parentListName));
        PANDORA_THROW_RESULT_IF(STATUS_CODE_SUCCESS, !=, this->GetClusterListName(pDaughterCluster, clusterListToNameMap, daughterListName));
        PANDORA_THROW_RESULT_IF(STATUS_CODE_SUCCESS, !=, PandoraContentApi::MergeAndDeleteClusters(*this, pParentCluster, pDaughterCluster,
            parentListName, daughterListName));
    }
    else
    {
        PANDORA_THROW_RESULT_IF(STATUS_CODE_SUCCESS, !=, PandoraContentApi::MergeAndDeleteClusters(*this, pParentCluster, pDaughterCluster));
    }
}

//------------------------------------------------------------------------------------------------------------------------------------------

StatusCode SoftClusterMergingAlgorithm::GetClusterListName(const Cluster *const pCluster, const ClusterListToNameMap &clusterListToNameMap,
    std::string &listName) const
{
    for (auto mapIter : clusterListToNameMap)
    {
        if (mapIter.first->end() != mapIter.first->find(pCluster))
        {
            listName = mapIter.second;
            return STATUS_CODE_SUCCESS;
        }
    }

    return STATUS_CODE_NOT_FOUND;
}

//------------------------------------------------------------------------------------------------------------------------------------------

StatusCode SoftClusterMergingAlgorithm::ReadSettings(const TiXmlHandle xmlHandle)
{
    PANDORA_RETURN_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=, XmlHelper::ReadValue(xmlHandle,
        "ShouldUseCurrentClusterList", m_shouldUseCurrentClusterList));

    PANDORA_RETURN_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=, XmlHelper::ReadValue(xmlHandle,
        "UpdateCurrentTrackClusterAssociations", m_updateCurrentTrackClusterAssociations));

    if (m_shouldUseCurrentClusterList && m_updateCurrentTrackClusterAssociations)
    {
        PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, XmlHelper::ProcessFirstAlgorithm(*this, xmlHandle, 
            m_trackClusterAssociationAlgName));
    }

    PANDORA_RETURN_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=, XmlHelper::ReadVectorOfValues(xmlHandle,
        "AdditionalClusterListNames", m_additionalClusterListNames));

    PANDORA_RETURN_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=, XmlHelper::ReadValue(xmlHandle,
        "MaxHitsInSoftCluster", m_maxHitsInSoftCluster));

    PANDORA_RETURN_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=, XmlHelper::ReadValue(xmlHandle,
        "MaxLayersSpannedBySoftCluster", m_maxLayersSpannedBySoftCluster));

    PANDORA_RETURN_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=, XmlHelper::ReadValue(xmlHandle,
        "MaxHadEnergyForSoftClusterNoTrack", m_maxHadEnergyForSoftClusterNoTrack));

    PANDORA_RETURN_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=, XmlHelper::ReadValue(xmlHandle,
        "MinClusterHadEnergy", m_minClusterHadEnergy));

    PANDORA_RETURN_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=, XmlHelper::ReadValue(xmlHandle,
        "MinClusterEMEnergy", m_minClusterEMEnergy));

    PANDORA_RETURN_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=, XmlHelper::ReadValue(xmlHandle,
        "MinCosOpeningAngle", m_minCosOpeningAngle));

    PANDORA_RETURN_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=, XmlHelper::ReadValue(xmlHandle,
        "MinHitsInCluster", m_minHitsInCluster));

    PANDORA_RETURN_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=, XmlHelper::ReadValue(xmlHandle,
        "ClosestDistanceCut0", m_closestDistanceCut0));

    PANDORA_RETURN_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=, XmlHelper::ReadValue(xmlHandle,
        "ClosestDistanceCut1", m_closestDistanceCut1));

    PANDORA_RETURN_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=, XmlHelper::ReadValue(xmlHandle,
        "InnerLayerCut1", m_innerLayerCut1));

    PANDORA_RETURN_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=, XmlHelper::ReadValue(xmlHandle,
        "ClosestDistanceCut2", m_closestDistanceCut2));

    PANDORA_RETURN_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=, XmlHelper::ReadValue(xmlHandle,
        "InnerLayerCut2", m_innerLayerCut2));

    PANDORA_RETURN_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=, XmlHelper::ReadValue(xmlHandle,
        "MaxClusterDistanceFine", m_maxClusterDistanceFine));

    PANDORA_RETURN_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=, XmlHelper::ReadValue(xmlHandle,
        "MaxClusterDistanceCoarse", m_maxClusterDistanceCoarse));

    return STATUS_CODE_SUCCESS;
}

} // namespace lc_content_fast
