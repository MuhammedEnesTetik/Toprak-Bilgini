import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';

void main() => runApp(const MyApp());

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Toprak Bilgini',
      theme: ThemeData(
        primarySwatch: Colors.green,
        appBarTheme: const AppBarTheme(
          color: Color(0xFF388E3C),
        ),
        fontFamily: 'Poppins',
        inputDecorationTheme: const InputDecorationTheme(
          filled: true,
          fillColor: Color(0xFFD7CCC8), // Açık kahverengi
          border: OutlineInputBorder(),
          focusedBorder: OutlineInputBorder(
            borderSide: BorderSide(color: Color(0xFF388E3C)), // Doğa yeşili
          ),
          enabledBorder: OutlineInputBorder(
            borderSide: BorderSide(color: Color(0xFF795548)), // Kahverengi
          ),
          labelStyle: TextStyle(color: Color(0xFF795548)), // Kahverengi
        ),
        elevatedButtonTheme: ElevatedButtonThemeData(
          style: ElevatedButton.styleFrom(
            backgroundColor: const Color(0xFF388E3C), // Doğa yeşili
            foregroundColor: Colors.white, // Buton yazı rengi
          ),
        ),
        textTheme: const TextTheme(
          bodyMedium: TextStyle(color: Colors.white), // Metin rengi
        ),
      ),
      home: const HomePage(),
    );
  }
}

class HomePage extends StatelessWidget {
  const HomePage({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Stack(
        fit: StackFit.expand,
        children: [
          Image.asset(
            'assets/t_anasayfa.jpeg', // Doğa temalı arka plan resmi
            fit: BoxFit.cover,
          ),
          Center(
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                const Text(
                  'Toprak Bilgini',
                  style: TextStyle(
                    fontSize: 36,
                    fontWeight: FontWeight.bold,
                    color: Colors.white,
                  ),
                ),
                const SizedBox(height: 220),
                ElevatedButton(
                  style: ElevatedButton.styleFrom(
                    shape: const CircleBorder(),
                    padding: const EdgeInsets.all(24),
                    backgroundColor: Colors.green, // Butonun arka plan rengi
                  ),
                  onPressed: () {
                    Navigator.push(
                      context,
                      MaterialPageRoute(
                          builder: (context) => const PredictionForm()),
                    );
                  },
                  child: const Icon(
                    Icons.spa,
                    color: Colors.white,
                    size: 50,
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

class PredictionForm extends StatefulWidget {
  const PredictionForm({super.key});

  @override
  // ignore: library_private_types_in_public_api
  _PredictionFormState createState() => _PredictionFormState();
}

class _PredictionFormState extends State<PredictionForm> {
  final _formKey = GlobalKey<FormState>();
  final TextEditingController _azotController = TextEditingController();
  final TextEditingController _fosforController = TextEditingController();
  final TextEditingController _potasyumController = TextEditingController();
  final TextEditingController _sicaklikController = TextEditingController();
  final TextEditingController _nemController = TextEditingController();
  final TextEditingController _phController = TextEditingController();
  final TextEditingController _yagisController = TextEditingController();

  String _result = '';
  bool _isLoading = false;

  Future<void> _submitForm() async {
    if (_formKey.currentState!.validate()) {
      setState(() {
        _isLoading = true;
      });

      final response = await http.post(
        Uri.parse('http://172.20.10.3:5000/predict'),
        headers: {'Content-Type': 'application/json'},
        body: json.encode({
          'Azot': double.parse(_azotController.text),
          'Fosfor': double.parse(_fosforController.text),
          'Potasyum': double.parse(_potasyumController.text),
          'Sıcaklık': double.parse(_sicaklikController.text),
          'Nem': double.parse(_nemController.text),
          'pH_Değeri': double.parse(_phController.text),
          'Yağış': double.parse(_yagisController.text),
        }),
      );

      if (kDebugMode) {
        print("API Response Status Code: ${response.statusCode}");
      }
      if (kDebugMode) {
        print("API Response Body: ${response.body}");
      }

      if (response.statusCode == 200) {
        final data = json.decode(response.body);
        if (kDebugMode) {
          print("Decoded Data: $data");
        }
        setState(() {
          _result =
              'Ürün: ${data['prediction']} - Uygunluk: ${data['probability']}%';
          _isLoading = false;
        });
      } else {
        setState(() {
          _result = 'Tahmin yapılamadı, lütfen tekrar deneyin.';
          _isLoading = false;
        });
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Toprak Analizi'),
        foregroundColor: Colors.white,
      ),
      body: Stack(
        fit: StackFit.expand,
        children: [
          Image.asset(
            'assets/t_anasayfa.jpeg', // Doğa temalı arka plan resmi
            fit: BoxFit.cover,
          ),
          Padding(
            padding: const EdgeInsets.all(16.0),
            child: Form(
              key: _formKey,
              child: ListView(
                children: <Widget>[
                  _buildTextField(_azotController, 'Azot'),
                  _buildTextField(_fosforController, 'Fosfor'),
                  _buildTextField(_potasyumController, 'Potasyum'),
                  _buildTextField(_sicaklikController, 'Sıcaklık'),
                  _buildTextField(_nemController, 'Nem'),
                  _buildTextField(_phController, 'pH Değeri'),
                  _buildTextField(_yagisController, 'Yağış'),
                  const SizedBox(height: 20),
                  _isLoading
                      ? const Center(child: CircularProgressIndicator())
                      : ElevatedButton(
                          onPressed: _submitForm,
                          child: const Text('Tahmin Et'),
                        ),
                  const SizedBox(height: 20),
                  Text(
                    _result,
                    style: const TextStyle(
                        fontSize: 20, fontWeight: FontWeight.bold),
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildTextField(TextEditingController controller, String label) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 16.0),
      child: TextFormField(
        controller: controller,
        decoration: InputDecoration(labelText: label),
        keyboardType: TextInputType.number,
        validator: (value) {
          if (value == null || value.isEmpty) {
            return '$label alanı boş olamaz';
          }
          if (double.tryParse(value) == null) {
            return 'Geçerli bir sayı girin';
          }
          return null;
        },
      ),
    );
  }
}
