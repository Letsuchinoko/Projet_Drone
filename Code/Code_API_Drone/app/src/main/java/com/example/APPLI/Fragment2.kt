package com.example.APPLI

import android.os.Bundle
import android.util.Log
import androidx.fragment.app.Fragment
import androidx.fragment.app.activityViewModels
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.TextView
import android.widget.Toast
import com.google.android.gms.maps.CameraUpdateFactory
import com.google.android.gms.maps.GoogleMap
import com.google.android.gms.maps.OnMapReadyCallback
import com.google.android.gms.maps.SupportMapFragment
import com.google.android.gms.maps.model.CameraPosition
import com.google.android.gms.maps.model.LatLng
import com.google.android.gms.maps.model.Marker
import com.google.android.gms.maps.model.MarkerOptions
import com.google.android.gms.maps.model.MapView


// TODO: Rename parameter arguments, choose names that match
// the fragment initialization parameters, e.g. ARG_ITEM_NUMBER
private const val ARG_PARAM1 = "param1"
private const val ARG_PARAM2 = "param2"

/**
 * A simple [Fragment] subclass.
 * Use the [Fragment2.newInstance] factory method to
 * create an instance of this fragment.
 */
class Fragment2 : Fragment(), OnMapReadyCallback {
    private val TAG = "Fragment2"
    // TODO: Rename and change types of parameters
    private var param1: String? = null
    private var param2: String? = null

    private lateinit var gpsTextView: TextView
    private lateinit var mapView: MapView
    private var googleMap: GoogleMap? = null
    private val sharedDataManager: SharedDataManager by activityViewModels()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        arguments?.let {
            param1 = it.getString(ARG_PARAM1)
            param2 = it.getString(ARG_PARAM2)
        }
    }

    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View? {
        Log.d(TAG, "Fragment2 onCreateView appelé")
        val view = inflater.inflate(R.layout.fragment_2, container, false)
        
        // Initialiser la vue
        gpsTextView = view.findViewById(R.id.textView2)
        gpsTextView.text = "Localisation GPS\n\nEn attente de données..."
        Log.d(TAG, "TextView GPS initialisé")
        
        // Observer les changements de données GPS
        sharedDataManager.gpsData.observe(viewLifecycleOwner) { gpsData ->
            Log.d(TAG, "Données GPS reçues dans Fragment2: '$gpsData'")
            gpsTextView.text = "Localisation GPS\n\nDonnées reçues:\n$gpsData"
            Log.d(TAG, "TextView GPS mis à jour")
        }
        
        mapView = view.findViewById(R.id.mapView)
        mapView.onCreate(savedInstanceState)
        mapView.getMapAsync(this)
        
        Log.d(TAG, "Fragment2 onCreateView terminé")
        return view
    }

    override fun onMapReady(map: GoogleMap) {
        googleMap = map
        // Observer les données GPS partagées
        sharedDataManager.gpsData.observe(viewLifecycleOwner) { gpsString ->
            // gpsString format: "lat,lon"
            val parts = gpsString.split(",")
            if (parts.size == 2) {
                val lat = parts[0].toDoubleOrNull()
                val lon = parts[1].toDoubleOrNull()
                if (lat != null && lon != null) {
                    val position = LatLng(lat, lon)
                    googleMap?.clear()
                    googleMap?.addMarker(MarkerOptions().position(position).title("Position reçue"))
                    googleMap?.moveCamera(CameraUpdateFactory.newLatLngZoom(position, 15f))
                }
            }
        }
    }

    // N'oublie pas de forwarder le cycle de vie à la MapView
    override fun onResume() { super.onResume(); mapView.onResume() }
    override fun onPause() { super.onPause(); mapView.onPause() }
    override fun onDestroy() { super.onDestroy(); mapView.onDestroy() }
    override fun onLowMemory() { super.onLowMemory(); mapView.onLowMemory() }

    companion object {
        /**
         * Use this factory method to create a new instance of
         * this fragment using the provided parameters.
         *
         * @param param1 Parameter 1.
         * @param param2 Parameter 2.
         * @return A new instance of fragment Fragment2.
         */
        // TODO: Rename and change types and number of parameters
        @JvmStatic
        fun newInstance(param1: String, param2: String) =
            Fragment2().apply {
                arguments = Bundle().apply {
                    putString(ARG_PARAM1, param1)
                    putString(ARG_PARAM2, param2)
                }
            }
    }
}